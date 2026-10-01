"""
MVBench Evaluation using tai_core
Usage:
    python eval_mvbench.py --mvbench_dir $MV \
      --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
      --tau_profile tau_norm_16f/tau_norm_Qwen3-VL-8B-Instruct.json \
      --beta $BETA --n_frames 16
"""
import torch, json, os, gc
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tai.tai_core import load_model_adapter, run_inference, get_token_ids, classify_from_logits, load_tau_profile

MVBENCH_DATA_LIST = {
    "Action Sequence":          ("action_sequence.json",          "video/star/Charades_v1_480/",           "video", True),
    "Action Prediction":        ("action_prediction.json",        "video/star/Charades_v1_480/",           "video", True),
    "Action Antonym":           ("action_antonym.json",           "video/ssv2_video/",                     "video", False),
    "Fine-grained Action":      ("fine_grained_action.json",      "video/Moments_in_Time_Raw/videos/",     "video", False),
    "Unexpected Action":        ("unexpected_action.json",        "video/FunQA_test/test/",                "video", False),
    "Object Existence":         ("object_existence.json",         "video/clevrer/video_validation/",       "video", False),
    "Object Interaction":       ("object_interaction.json",       "video/star/Charades_v1_480/",           "video", True),
    "Object Shuffle":           ("object_shuffle.json",           "video/perception/videos/",              "video", False),
    "Moving Direction":         ("moving_direction.json",         "video/clevrer/video_validation/",       "video", False),
    "Action Localization":      ("action_localization.json",      "video/sta/sta_video/",                  "video", True),
    "Scene Transition":         ("scene_transition.json",         "video/scene_qa/video/",                 "video", False),
    "Action Count":             ("action_count.json",             "video/perception/videos/",              "video", False),
    "Moving Count":             ("moving_count.json",             "video/clevrer/video_validation/",       "video", False),
    "Moving Attribute":         ("moving_attribute.json",         "video/clevrer/video_validation/",       "video", False),
    "State Change":             ("state_change.json",             "video/perception/videos/",              "video", False),
    "Fine-grained Pose":        ("fine_grained_pose.json",        "video/nturgbd/",                        "video", False),
    "Character Order":          ("character_order.json",          "video/perception/videos/",              "video", False),
    "Egocentric Navigation":    ("egocentric_navigation.json",    "video/vlnqa/",                          "video", False),
    "Episodic Reasoning":       ("episodic_reasoning.json",       "video/tvqa/frames_fps3_hq/",            "frame", True),
    "Counterfactual Inference": ("counterfactual_inference.json", "video/clevrer/video_validation/",       "video", False),
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

def load_mvbench_data(mvbench_dir):
    mvbench_dir = Path(mvbench_dir); json_dir = mvbench_dir / "json"; all_data = []; task_counts = {}
    for task_name, (json_file, video_subdir, data_type, _) in MVBENCH_DATA_LIST.items():
        json_path = json_dir / json_file
        if not json_path.exists(): continue
        with open(json_path) as f: items = json.load(f)
        video_prefix = str(mvbench_dir / video_subdir)
        for item in items:
            video_path = os.path.join(video_prefix, item["video"])
            if not os.path.exists(video_path):
                for ext in [".mp4",".mkv",".avi",".webm"]:
                    if os.path.exists(video_path + ext): video_path += ext; break
                else: continue
            all_data.append({"task": task_name, "video_path": video_path, "data_type": data_type, "data": item})
        task_counts[task_name] = len(items)
    print(f"  Loaded {len(all_data)} questions across {len(task_counts)} tasks"); return all_data

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvbench_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "tai"])
    parser.add_argument("--output_dir", type=str, default="results_mvbench")
    parser.add_argument("--src_layer", type=int, default=-1)
    parser.add_argument("--inject_start", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--tau_profile", type=str, default="")
    parser.add_argument("--clip_last", type=int, default=0, help="Clip last N layers to 0 in tau profile")
    parser.add_argument("--inverse_tau", action="store_true", help="Inverse mode: weight = 1 - normalized_tau")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output already exists")
    args = parser.parse_args()

    adapter = load_model_adapter(args.model_name)
    n_layers = adapter.n_layers()
    tau_profile = None
    if args.tau_profile and args.method == "tai":
        tau_profile, auto_src = load_tau_profile(args.tau_profile, clip_last=args.clip_last, inverse=args.inverse_tau)
        if args.src_layer < 0: args.src_layer = auto_src
        else: print(f"  src_layer override: L{args.src_layer} (profile recommended=L{auto_src})")
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
    print(f"\nConfig: {method_name}, layers={n_layers}")
    if args.skip_existing:
        _check = Path(args.output_dir) / f"{method_name}.json"
        if _check.exists():
            print(f"  SKIP (already complete): {method_name}")
            return

    token_ids = get_token_ids(adapter.tokenizer, "ABCDE")
    all_data = load_mvbench_data(args.mvbench_dir)
    task_stats = {}; results = []; skipped = 0
    for item in tqdm(all_data, desc="MVBench"):
        if item["data_type"] == "frame": skipped += 1; continue
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

    print(f"\nMVBench: {method_name}")
    tc, ta = 0, 0
    for t in sorted(task_stats):
        s = task_stats[t]; acc = 100*s["correct"]/s["total"] if s["total"] else 0
        print(f"  {t:30s}: {s['correct']:>4d}/{s['total']:<4d} ({acc:5.1f}%)")
        tc += s["correct"]; ta += s["total"]
    if ta: print(f"  {'TOTAL':30s}: {tc:>4d}/{ta:<4d} ({100*tc/ta:5.1f}%)")
    if skipped: print(f"  Skipped (frame type): {skipped}")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{method_name}.json", "w") as f: json.dump(results, f, indent=2)
    print(f"  Saved: {out_dir / method_name}.json")

if __name__ == "__main__":
    main()
