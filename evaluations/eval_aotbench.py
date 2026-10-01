"""
AoTBench Unified Evaluation via tai_core
==========================================
5 tasks (QA, ReverseFilm, UCF101, Rtime_t2v, Rtime_v2t)
Methods: baseline, tai (linear decay or τ-guided, with optional inverse/clip)
Model families: Qwen2.5-VL (incl. ArrowRL), Qwen3-VL, InternVL2.5/3, Molmo2, Gemma4

Usage:
    AOT=$DATA_ROOT/AoTBench
    JSONS="ReverseFilm UCF101 Rtime_t2v Rtime_v2t AoTBench_QA"

    # baseline
    for j in $JSONS; do
        python eval_aotbench.py --json_path $AOT/data_files/${j}.json --video_dir $AOT \
            --model_name Qwen/Qwen3-VL-8B-Instruct --method baseline --output_dir aot_q3
    done

    # τ-guided (src layer auto-selected from the profile peak)
    for j in $JSONS; do
        python eval_aotbench.py --json_path $AOT/data_files/${j}.json --video_dir $AOT \
            --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
            --tau_profile tau_norm_16f/tau_norm_Qwen3-VL-8B-Instruct.json \
            --beta $BETA --output_dir aot_q3
    done

    # Linear decay (no profile; manual src/inject)
    python eval_aotbench.py --json_path $AOT/data_files/AoTBench_QA.json --video_dir $AOT \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --method tai \
        --src_layer $SRC --inject_start $INJ --beta $BETA --output_dir aot_q25

    # Inverse guided (counter-hypothesis sanity check)
    python eval_aotbench.py --json_path $AOT/data_files/AoTBench_QA.json --video_dir $AOT \
        --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
        --tau_profile tau_norm_16f/tau_norm_Qwen3-VL-8B-Instruct.json \
        --inverse_tau --beta $BETA --output_dir aot_q3

    # Anti-TAI (negative β, steering test)
    python eval_aotbench.py --json_path $AOT/data_files/AoTBench_QA.json --video_dir $AOT \
        --model_name Qwen/Qwen3-VL-8B-Instruct --method tai \
        --tau_profile tau_norm_16f/tau_norm_Qwen3-VL-8B-Instruct.json \
        --beta -$BETA --output_dir aot_q3
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
from tai.tai_core import (
    load_model_adapter, run_inference, load_tau_profile,
    get_token_ids, classify_from_logits, prepare_input
)


# ============================================================
# Free-form generation (for tasks without answer_list)
# ============================================================

@torch.no_grad()
def generate_free_form(adapter, video_path, prompt, n_frames=16, max_new_tokens=10):
    """Greedy generate for free-form QA. Returns first character (uppercased)."""
    inputs, _, _ = prepare_input(adapter, video_path, prompt, n_frames=n_frames)

    if adapter.family == "internvl" and "inputs_embeds" in inputs:
        gen_ids = adapter.model.language_model.generate(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens, do_sample=False,
        )
        text = adapter.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    else:
        gen_ids = adapter.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if adapter.processor is not None:
            in_ids = inputs.get("input_ids")
            if in_ids is not None:
                trimmed = [out[len(in_):] for in_, out in zip(in_ids, gen_ids)]
                text = adapter.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            else:
                text = adapter.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        else:
            text = adapter.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    del inputs
    torch.cuda.empty_cache()
    return text[0].upper() if len(text) > 0 else ""


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="aot_results")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "tai"])

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
                        help="Inverse mode: weight = 1 - normalized_tau")
    parser.add_argument("--src_window", type=int, default=1,
                        help="Window size for src extraction (1=single, 3=avg of 3 layers)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output already exists")
    args = parser.parse_args()

    # ----- Adapter -----
    adapter = load_model_adapter(args.model_name)
    n_layers = adapter.n_layers()

    # ----- τ profile -----
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

    # ----- Method name -----
    task_name = Path(args.json_path).stem
    model_short = args.model_name.split("/")[-1]

    if args.method == "baseline":
        method_suffix = "baseline"
    else:
        if tau_profile is not None:
            tag = "invguided" if args.inverse_tau else "guided"
        else:
            tag = "linear"
        beta_tag = f"b{args.beta}"
        if args.beta < 0:
            beta_tag = f"anti_b{abs(args.beta)}"
        method_suffix = f"tai_{tag}_src{args.src_layer}"
        if tau_profile is None:
            method_suffix += f"_inj{args.inject_start}"
        elif args.inject_start > args.src_layer:
            method_suffix += f"_inj{args.inject_start}"
        method_suffix += f"_{beta_tag}"
        if args.clip_last > 0:
            method_suffix += f"_clip{args.clip_last}"
        if args.src_window > 1:
            method_suffix += f"_win{args.src_window}"
    method_suffix += f"_nf{args.n_frames}"

    print(f"\nConfig: AoT_{task_name} | {model_short} | {method_suffix}")

    # Skip if already done
    if args.skip_existing:
        _check = Path(args.output_dir) / f"AoT_{task_name}_{model_short}_{method_suffix}.json"
        if _check.exists():
            print(f"  SKIP (already complete): AoT_{task_name}_{method_suffix}")
            return

    # ----- Data -----
    with open(args.json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"  Loaded {len(qa_data)} questions")

    video_map = {}
    for ext in ["*.mp4", "*.mkv", "*.avi", "*.webm"]:
        for p in Path(args.video_dir).rglob(ext):
            video_map[p.name] = str(p)
    print(f"  Indexed {len(video_map)} videos")

    # ----- Run -----
    correct, total = 0, 0
    failed = []

    for item in tqdm(qa_data, desc=task_name):
        raw_name = Path(item["video_name"]).name
        video_path = video_map.get(raw_name)
        if not video_path:
            item["predict"] = "error: missing video"
            failed.append(item["video_name"])
            continue

        prompt = item["question"]

        valid_choices = []
        if "answer_list" in item:
            valid_choices = [chr(ord('A') + i) for i in range(len(item['answer_list']))]
        elif str(item.get("ans", "")).strip().upper() in ["A", "B", "C", "D", "E"]:
            valid_choices = ["A", "B", "C", "D", "E"]

        try:
            if valid_choices:
                logits = run_inference(
                    adapter, video_path, prompt,
                    args.method, args.src_layer, args.inject_start, args.beta, args.n_frames,
                    tau_profile=tau_profile, src_window=args.src_window,
                )
                token_ids = get_token_ids(adapter.tokenizer, "ABCDE")
                pred = classify_from_logits(logits, token_ids, valid_choices)
            else:
                # Free-form generate (baseline path; no TAI integration for generate)
                pred = generate_free_form(adapter, video_path, prompt, n_frames=args.n_frames)

            item["predict"] = pred
            ans = str(item.get("ans", "")).strip().upper()
            if ans == pred:
                correct += 1
            total += 1

        except Exception as e:
            print(f"\n[Error] {item['video_name']}: {e}")
            item["predict"] = f"error: {e}"
            failed.append(item["video_name"])

        if total > 0 and total % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ----- Save & Print -----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"AoT_{task_name}_{model_short}_{method_suffix}.json"
    with open(out_file, "w", encoding='utf-8') as f:
        json.dump(qa_data, f, indent=2)

    acc = (correct / total) * 100 if total > 0 else 0
    print(f"\nAoT_{task_name} | {method_suffix}: {correct}/{total} ({acc:.2f}%)")
    if failed:
        print(f"  failed: {len(failed)}")
    print(f"  saved: {out_file}")


if __name__ == "__main__":
    main()
