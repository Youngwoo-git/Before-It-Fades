"""
TempCompass Evaluation via tai_core
====================================
Methods: baseline, tai (+β), anti-tai (-β)
+ --sample_seed: frame-sampling jitter for robustness checks (None = uniform sampling)

Usage:
    # baseline
    python eval_tempcompass.py --video_dir $TC_V --data_dir $TC_D \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --method baseline

    # τ-guided TAI
    python eval_tempcompass.py --video_dir $TC_V --data_dir $TC_D \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --method tai \
        --tau_profile tau_norm_16f/tau_norm_Qwen2.5-VL-7B-Instruct.json --beta $BETA

    # frame-sampling jitter run
    python eval_tempcompass.py --video_dir $TC_V --data_dir $TC_D \
        --output_dir tc_seed_jitter --sample_seed 0 \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --method baseline
"""

import torch
import json
import os
import gc
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tai.tai_core import load_model_adapter, run_inference, load_tau_profile

# ============================================================
# TempCompass-specific classification
# ============================================================

ANSWER_PROMPTS = {
    "yes_no": "\nPlease answer yes or no:",
    "caption_matching": "\nPlease directly give the best option:",
    "multi-choice": "\nPlease directly give the best option:",
}


def get_yn_token_ids(tokenizer):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)[0]
    return {
        "yes": encode("yes"), "no": encode("no"),
        "Yes": encode("Yes"), "No": encode("No"),
        "A": encode("A"), "B": encode("B"),
        "C": encode("C"), "D": encode("D"),
        "1": encode("1"), "2": encode("2"),
    }


def classify_yes_no(logits, tids):
    y = torch.logsumexp(
        torch.tensor([logits[tids["yes"]].item(), logits[tids["Yes"]].item()]), 0).item()
    n = torch.logsumexp(
        torch.tensor([logits[tids["no"]].item(), logits[tids["No"]].item()]), 0).item()
    return "yes" if y > n else "no"


def classify_caption_matching(logits, tids, q):
    if "Option 1:" in q and "Option 2:" in q:
        return "Option 1" if logits[tids["1"]].item() > logits[tids["2"]].item() else "Option 2"
    la, lb = logits[tids["A"]].item(), logits[tids["B"]].item()
    if "Sentence A:" in q:
        return "Sentence A" if la > lb else "Sentence B"
    elif "Caption A:" in q:
        return "Caption A" if la > lb else "Caption B"
    return "A" if la > lb else "B"


def classify_multi_choice(logits, tids, q):
    has_d = any(x in q for x in ["D.", "D)", "\nD.", "\nD "])
    cands = ["A", "B", "C"] + (["D"] if has_d else [])
    scores = {c: logits[tids[c]].item() for c in cands}
    return max(scores, key=scores.get)


def classify(logits, tids, q, fmt):
    if fmt == "yes_no":
        return classify_yes_no(logits, tids)
    elif fmt == "caption_matching":
        return classify_caption_matching(logits, tids, q)
    elif fmt == "multi-choice":
        return classify_multi_choice(logits, tids, q)


def is_correct(pred, fmt):
    p, a = pred["prediction"], pred["answer"]
    if p is None:
        return False
    if fmt == "yes_no":
        return p.lower() == a.lower()
    elif fmt == "caption_matching":
        return p in a or p == a.split(":")[0].strip()
    elif fmt == "multi-choice":
        return p == a[0]
    return False


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="TempCompass Eval via tai_core")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "tai"])
    parser.add_argument("--output_dir", type=str, default="tc_steering")
    parser.add_argument("--formats", type=str, nargs="+",
                        default=["yes_no", "caption_matching", "multi-choice"])
    # TAI params
    parser.add_argument("--src_layer", type=int, default=-1)
    parser.add_argument("--inject_start", type=int, default=-1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--tau_profile", type=str, default="",
                        help="τ norm JSON for guided injection")
    parser.add_argument("--clip_last", type=int, default=0,
                        help="Clip last N layers to 0 in tau profile")
    parser.add_argument("--inverse_tau", action="store_true",
                        help="Inverse mode: weight = 1 - normalized_tau (counter-hypothesis)")
    parser.add_argument("--src_window", type=int, default=1,
                        help="Window size for src extraction (1=single layer, 3=avg of src-1,src,src+1)")
    parser.add_argument("--sample_seed", type=int, default=None,
                        help="Frame-sampling jitter seed (None = legacy uniform sampling)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output already exists")
    args = parser.parse_args()

    adapter = load_model_adapter(args.model_name)
    n_layers = adapter.n_layers()
    tids = get_yn_token_ids(adapter.tokenizer)

    # Load tau profile if provided
    tau_profile = None
    if args.tau_profile and args.method == "tai":
        tau_profile, auto_src = load_tau_profile(
            args.tau_profile, clip_last=args.clip_last, inverse=args.inverse_tau)
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
        if args.inject_start < 0:
            args.inject_start = min(args.src_layer + 1, n_layers - 1)

    # Method name for output
    model_short = args.model_name.split("/")[-1]
    if args.method == "baseline":
        method_name = f"{model_short}_baseline"
    else:
        if tau_profile is not None:
            tag = "invguided" if args.inverse_tau else "guided"
        else:
            tag = "linear"
        beta_tag = f"b{args.beta}"
        if args.beta < 0:
            beta_tag = f"anti_b{abs(args.beta)}"
        method_name = f"{model_short}_tai_{tag}_src{args.src_layer}"
        # inject_start tag: linear always; guided only if overridden past src+1
        if tau_profile is None:
            method_name += f"_inj{args.inject_start}"
        elif args.inject_start > args.src_layer:
            method_name += f"_inj{args.inject_start}"
        method_name += f"_{beta_tag}"
        if args.clip_last > 0:
            method_name += f"_clip{args.clip_last}"
        if args.src_window > 1:
            method_name += f"_win{args.src_window}"
    method_name += f"_nf{args.n_frames}"
    if args.sample_seed is not None:
        method_name += f"_seed{args.sample_seed}"

    print(f"\nConfig: {method_name}")

    # Skip if already done
    if args.skip_existing:
        out_check = Path(args.output_dir) / method_name
        done = [f for f in args.formats if (out_check / f"{f}.json").exists()]
        if len(done) == len(args.formats):
            print(f"  SKIP (already complete): {method_name}")
            return

    # Load data
    data_dir = Path(args.data_dir)
    qa_data = {}
    for fmt in args.formats:
        fp = data_dir / f"{fmt}.json"
        if fp.exists():
            with open(fp) as f:
                qa_data[fmt] = json.load(f)
            print(f"  Loaded {fmt}: {len(qa_data[fmt])} videos")

    out_dir = Path(args.output_dir) / method_name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)

    # Evaluate
    for fmt, videos_data in qa_data.items():
        print(f"\n=== {fmt} ({len(videos_data)} videos) ===")
        predictions = {}

        for video_id, categories in tqdm(videos_data.items(), desc=fmt):
            vp = video_dir / f"{video_id}.mp4"
            if not vp.exists():
                continue
            predictions[video_id] = {}

            for category, questions in categories.items():
                cat_results = []
                for qa in questions:
                    question, answer = qa["question"], qa["answer"]
                    prompt = question + ANSWER_PROMPTS[fmt]

                    try:
                        logits = run_inference(
                            adapter, str(vp), prompt,
                            args.method, args.src_layer, args.inject_start,
                            args.beta, args.n_frames,
                            tau_profile=tau_profile, src_window=args.src_window,
                            sample_seed=args.sample_seed)
                        prediction = classify(logits, tids, question, fmt)
                    except Exception as e:
                        print(f"  Error {video_id}: {e}")
                        prediction = None

                    cat_results.append({
                        "question": question,
                        "answer": answer,
                        "prediction": prediction,
                    })

                predictions[video_id][category] = cat_results

        # Save
        out_path = out_dir / f"{fmt}.json"
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)

        # Accuracy
        total_c, total_n = 0, 0
        per_cat = {}
        for vid, cats in predictions.items():
            for cat, preds in cats.items():
                if cat not in per_cat:
                    per_cat[cat] = {"correct": 0, "total": 0}
                for p in preds:
                    per_cat[cat]["total"] += 1
                    total_n += 1
                    if p["prediction"] and is_correct(p, fmt):
                        per_cat[cat]["correct"] += 1
                        total_c += 1

        print(f"\n--- {fmt} ---")
        for cat in sorted(per_cat):
            c = per_cat[cat]
            acc = 100 * c["correct"] / c["total"] if c["total"] else 0
            print(f"  {cat:20s}: {c['correct']:>4d}/{c['total']:<4d} ({acc:5.1f}%)")
        if total_n:
            print(f"  {'TOTAL':20s}: {total_c:>4d}/{total_n:<4d} ({100*total_c/total_n:5.1f}%)")

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone: {method_name} → {out_dir}")


if __name__ == "__main__":
    main()