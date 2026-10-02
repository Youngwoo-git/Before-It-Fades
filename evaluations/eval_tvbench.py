"""
TVBench Evaluation using tai_core (τ norm-guided injection supported)
Usage:
    # τ norm-guided (auto src from profile)
    python eval_tvbench.py --tvbench_dir $TV \
      --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
      --tau_profile tau_norm_16f/tau_norm_Qwen3-VL-8B-Instruct.json \
      --beta $BETA --n_frames 16

    # τ norm-guided with src override
    python eval_tvbench.py --tvbench_dir $TV \
      --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
      --tau_profile tau_norm_16f/tau_norm_Qwen3-VL-8B-Instruct.json \
      --src_layer $SRC --beta $BETA --n_frames 16

    # Linear decay (no profile; manual src/inject)
    python eval_tvbench.py --tvbench_dir $TV \
      --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
      --src_layer $SRC --inject_start $INJ --beta $BETA --n_frames 16
"""
import torch, json, os, gc
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tai.tai_core import load_model_adapter, run_inference, get_token_ids, classify_from_logits, load_tau_profile

TVBENCH_TASKS = {
    "Action Count":        ("action_count.json",        "video/action_count"),
    "Action Localization": ("action_localization.json",  "video/action_localization"),
    "Action Sequence":     ("action_sequence.json",      "video/action_sequence"),
    "Egocentric Sequence": ("egocentric_sequence.json",  "video/egocentric_sequence"),
    "Moving Direction":    ("moving_direction.json",     "video/moving_direction"),
    "Object Count":        ("object_count.json",         "video/object_count"),
    "Object Shuffle":      ("object_shuffle.json",       "video/object_shuffle"),
    "Scene Transition":    ("scene_transition.json",     "video/scene_transition"),
    "Unexpected Action":   ("unexpected_action.json",    "video/unexpected_action"),
}
SYSTEM_PROMPT = ("Carefully watch the video and pay attention to the cause and sequence of events, "
                 "the detail and movement of objects, and the action and pose of persons.")
ANSWER_PROMPT = "\nPlease directly give the best option:"

def format_question(data):
    question = f"Question: {data['question']}\nOptions:\n"
    answer_text = data["answer"]; answer_idx = -1
    for idx, c in enumerate(data["candidates"]):
        letter = chr(ord("A") + idx); question += f"({letter}) {c}\n"
        if c == answer_text: answer_idx = idx
    return question.rstrip(), chr(ord("A") + answer_idx) if answer_idx >= 0 else "A", len(data["candidates"])

def find_video(name, video_dir):
    direct = os.path.join(video_dir, name)
    if os.path.exists(direct): return direct
    for root, dirs, files in os.walk(video_dir):
        if name in files: return os.path.join(root, name)
    return None

def load_tvbench_data(tvbench_dir):
    tvbench_dir = Path(tvbench_dir); json_dir = tvbench_dir / "json"; all_data = []
    for task_name, (jf, vd) in TVBENCH_TASKS.items():
        jp = json_dir / jf
        if not jp.exists(): continue
        with open(jp) as f: items = json.load(f)
        vb = str(tvbench_dir / vd)
        for item in items:
            vp = find_video(item["video"], vb)
            if not vp: vp = find_video(item["video"], str(tvbench_dir / "video"))
            if vp: all_data.append({"task": task_name, "video_path": vp, "data": item})
    print(f"  Loaded {len(all_data)} items"); return all_data

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvbench_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "tai"])
    parser.add_argument("--output_dir", type=str, default="results_tvbench")
    parser.add_argument("--src_layer", type=int, default=-1)
    parser.add_argument("--inject_start", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--tau_profile", type=str, default="", help="τ norm JSON → guided injection")
    parser.add_argument("--clip_last", type=int, default=0, help="Clip last N layers to 0 in tau profile")
    parser.add_argument("--inverse_tau", action="store_true", help="Inverse mode: weight = 1 - normalized_tau")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output already exists")
    args = parser.parse_args()

    adapter = load_model_adapter(args.model_name)
    n_layers = adapter.n_layers()
    tau_profile = None
    if args.tau_profile and args.method == "tai":
        tau_profile, auto_src = load_tau_profile(args.tau_profile, clip_last=args.clip_last, inverse=args.inverse_tau)
        # Use the profile peak as src_layer unless --src_layer is given explicitly
        if args.src_layer < 0:
            args.src_layer = auto_src
        else:
            print(f"  src_layer override: L{args.src_layer} (profile recommended=L{auto_src})")
    elif args.method == "tai":
        if args.src_layer < 0:
            raise SystemExit(
                "TAI is τ-profile-guided: pass --tau_profile "
                "(extract one with tau_profile/tau_norm_multi_model.py), "
                "or set --src_layer explicitly for a linear-decay run.")
        if args.inject_start < 0: args.inject_start = min(args.src_layer + 1, n_layers - 1)

    model_short = args.model_name.split("/")[-1]
    method_name = f"{model_short}_{args.method}"
    if args.method == "tai":
        if tau_profile is not None:
            tag = "invguided" if args.inverse_tau else "guided"
        else:
            tag = "linear"
        beta_tag = f"b{args.beta}"
        if args.beta < 0:
            beta_tag = f"anti_b{abs(args.beta)}"
        method_name += f"_{tag}_src{args.src_layer}"
        if tau_profile is None:
            method_name += f"_inj{args.inject_start}"
        elif args.inject_start > args.src_layer:
            method_name += f"_inj{args.inject_start}"
        method_name += f"_{beta_tag}"
        if args.clip_last > 0: method_name += f"_clip{args.clip_last}"
    method_name += f"_nf{args.n_frames}"
    print(f"\nConfig: {method_name}")
    if args.skip_existing:
        _check = Path(args.output_dir) / f"{method_name}.json"
        if _check.exists():
            print(f"  SKIP (already complete): {method_name}")
            return

    token_ids = get_token_ids(adapter.tokenizer, "ABCDE")
    all_data = load_tvbench_data(args.tvbench_dir)
    task_stats = {}; results = []
    for item in tqdm(all_data, desc="TVBench"):
        task = item["task"]
        question, answer_letter, n_cands = format_question(item["data"])
        prompt = SYSTEM_PROMPT + "\n" + question + ANSWER_PROMPT
        try:
            logits = run_inference(adapter, item["video_path"], prompt, args.method,
                                   args.src_layer, args.inject_start, args.beta, args.n_frames,
                                   tau_profile=tau_profile)
            cands = [chr(ord("A")+i) for i in range(n_cands)]
            pred = classify_from_logits(logits, token_ids, cands)
            correct = (pred == answer_letter)
        except Exception as e:
            print(f"\n  Error: {e}"); pred = "error"; correct = False
        if task not in task_stats: task_stats[task] = {"correct": 0, "total": 0}
        task_stats[task]["total"] += 1
        if correct: task_stats[task]["correct"] += 1
        results.append({"task": task, "answer": answer_letter, "prediction": pred, "correct": correct})
        if len(results) % 100 == 0: gc.collect()

    print(f"\nTVBench: {method_name}")
    tc, ta = 0, 0
    for t in sorted(task_stats):
        s = task_stats[t]; acc = 100*s["correct"]/s["total"] if s["total"] else 0
        print(f"  {t:25s}: {s['correct']:>4d}/{s['total']:<4d} ({acc:5.1f}%)")
        tc += s["correct"]; ta += s["total"]
    if ta: print(f"  {'TOTAL':25s}: {tc:>4d}/{ta:<4d} ({100*tc/ta:5.1f}%)")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{method_name}.json", "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {out_dir / method_name}.json")

if __name__ == "__main__":
    main()
