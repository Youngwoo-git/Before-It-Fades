"""
Multi-Model τ Norm Analysis: Proving Universality of Mid-Layer Temporal Signal
===============================================================================
Supports: Qwen2.5-VL, Qwen3-VL, InternVL2.5, InternVL3, Molmo2, Gemma4
(ArrowRL-Qwen2.5-VL is handled through the Qwen family path.)

Measures ||h_fwd - h_rev|| / ||h_fwd|| per layer to show temporal signal peaks
at mid-to-late layers across ALL model families. Outputs per-model plots and a
JSON τ profile consumed by the eval scripts (--tau_profile).

Usage:
    python tau_norm_multi_model.py \
        --video_dir $TC_V --qa_json $TC_D/yes_no.json \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct --max_pairs 50 \
        --n_frames 16 --output_dir tau_norm_16f

Smoke-test a new model with --max_pairs 1 first.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import json, gc
from pathlib import Path
from tqdm import tqdm
import argparse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tai.tai_core import load_model_adapter, prepare_input

# ============================================================
# 2. Data (TempCompass direction pairs)
# ============================================================

def load_direction_pairs(qa_json_path, video_dir):
    with open(qa_json_path) as f:
        qa_data = json.load(f)
    video_dir = Path(video_dir)
    entries = []
    for video_key, categories in qa_data.items():
        if "direction" not in categories: continue
        video_file = video_dir / f"{video_key}.mp4"
        if not video_file.exists(): continue
        entries.append({
            "key": video_key, "path": str(video_file),
            "is_reverse": "_reverse" in video_key,
            "questions": categories["direction"]})
    fwd = {e["key"]: e for e in entries if not e["is_reverse"]}
    rev = {e["key"].replace("_reverse", ""): e for e in entries if e["is_reverse"]}
    pairs = []
    for k in sorted(fwd):
        if k in rev:
            pairs.append({"name": k, "forward": fwd[k], "reverse": rev[k]})
    print(f"Loaded {len(pairs)} direction pairs")
    return pairs


# ============================================================
# 3. Per-layer hidden state extraction (all layers, one pass)
# ============================================================
# Model loading and input preparation are shared with the eval scripts
# (tai/tai_core.py), so the profile is measured on exactly the inputs TAI sees.

@torch.no_grad()
def extract_all_layer_hiddens(adapter, inputs):
    """Extract last-token hidden state at every layer. Returns list of (dim,) CPU tensors."""
    hiddens = []

    def hook_fn(module, input, output):
        h = (output[0] if isinstance(output, tuple) else output)[0, -1, :].detach().float().cpu()
        hiddens.append(h)

    handles = [layer.register_forward_hook(hook_fn) for layer in adapter.get_layers()]
    adapter.patch_lm_head()
    try:
        adapter.forward_model(inputs)
    finally:
        adapter.unpatch_lm_head()
        for h in handles: h.remove()

    torch.cuda.empty_cache()
    return hiddens


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--qa_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_pairs", type=int, default=50)
    parser.add_argument("--n_frames", type=int, default=16, help="Number of frames to sample from video")
    parser.add_argument("--output_dir", type=str, default="tau_norm_plots")
    args = parser.parse_args()

    adapter = load_model_adapter(args.model_name)
    n_layers = adapter.n_layers()
    model_short = args.model_name.rstrip("/").split("/")[-1]
    tag = f"_nf{args.n_frames}" if args.n_frames != 16 else ""

    print(f"  {n_layers} layers, family={adapter.family}, n_frames={args.n_frames}")

    pairs = load_direction_pairs(args.qa_json, args.video_dir)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    all_results = []

    for pair in tqdm(pairs, desc=f"tau norms ({model_short})"):
        question = pair["forward"]["questions"][0]["question"]
        prompt = f"{question} Answer with only one word: yes or no."
        try:
            inputs_fwd, _, _ = prepare_input(adapter, pair["forward"]["path"], prompt, n_frames=args.n_frames)
            h_fwd = extract_all_layer_hiddens(adapter, inputs_fwd)
            del inputs_fwd; torch.cuda.empty_cache()

            inputs_rev, _, _ = prepare_input(adapter, pair["reverse"]["path"], prompt, n_frames=args.n_frames)
            h_rev = extract_all_layer_hiddens(adapter, inputs_rev)
            del inputs_rev; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Skip {pair['name']}: {e}")
            torch.cuda.empty_cache()
            continue

        if len(h_fwd) != n_layers or len(h_rev) != n_layers:
            # Detect architectures where hooks fire more than once per layer (recursive forward etc.)
            print(f"  Skip {pair['name']}: hook count mismatch "
                  f"(fwd={len(h_fwd)}, rev={len(h_rev)}, layers={n_layers})")
            continue

        tau_norms = []
        h_norms = []
        cos_sims = []
        for l in range(n_layers):
            tau = h_fwd[l] - h_rev[l]
            tau_norms.append(torch.norm(tau).item())
            h_norms.append(torch.norm(h_fwd[l]).item())
            cos = torch.nn.functional.cosine_similarity(h_fwd[l].unsqueeze(0), h_rev[l].unsqueeze(0)).item()
            cos_sims.append(cos)

        all_results.append({
            "name": pair["name"],
            "tau_norms": tau_norms,
            "h_norms": h_norms,
            "cos_sims": cos_sims,
        })

        if len(all_results) % 10 == 0:
            gc.collect()

    del adapter; gc.collect(); torch.cuda.empty_cache()

    n = len(all_results)
    if n == 0:
        print("No results!"); return

    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    tau_norms = np.array([r["tau_norms"] for r in all_results])
    h_norms = np.array([r["h_norms"] for r in all_results])
    rel_tau = tau_norms / (h_norms + 1e-8)
    cos_sims = np.array([r["cos_sims"] for r in all_results])

    x = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    mean, std = tau_norms.mean(0), tau_norms.std(0)
    ax.plot(x, mean, color="purple", lw=2.5)
    ax.fill_between(x, mean-std, mean+std, alpha=0.15, color="purple")
    peak = np.argmax(mean)
    ax.scatter([peak], [mean[peak]], color="red", s=100, zorder=5)
    ax.annotate(f"Peak: L{peak} ({mean[peak]:.1f})", (peak, mean[peak]),
                textcoords="offset points", xytext=(10,10), fontsize=11, color="red", fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel("||tau|| = ||h(V) - h(V_rev)||", fontsize=12)
    ax.set_title("Temporal Steering Vector Magnitude", fontsize=13); ax.grid(True, alpha=0.3)

    ax = axes[1]
    mean_r, std_r = rel_tau.mean(0), rel_tau.std(0)
    ax.plot(x, mean_r, color="teal", lw=2.5)
    ax.fill_between(x, mean_r-std_r, mean_r+std_r, alpha=0.15, color="teal")
    peak_r = np.argmax(mean_r)
    ax.scatter([peak_r], [mean_r[peak_r]], color="red", s=100, zorder=5)
    ax.annotate(f"Peak: L{peak_r} ({mean_r[peak_r]:.3f})", (peak_r, mean_r[peak_r]),
                textcoords="offset points", xytext=(10,10), fontsize=11, color="red", fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel("||tau|| / ||h|| (relative)", fontsize=12)
    ax.set_title("Normalized Temporal Signal", fontsize=13); ax.grid(True, alpha=0.3)

    plt.suptitle(f"{model_short}: Activation-Level Temporal Signal ({n} pairs)", fontsize=14)
    plt.tight_layout()
    fig.savefig(save_dir / f"tau_norm_{model_short}{tag}.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / f"tau_norm_{model_short}{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: tau_norm_{model_short}{tag}.png/pdf")

    fig, ax = plt.subplots(figsize=(12, 5))
    mean_c, std_c = cos_sims.mean(0), cos_sims.std(0)
    ax.plot(x, mean_c, color="darkorange", lw=2.5)
    ax.fill_between(x, mean_c-std_c, mean_c+std_c, alpha=0.15, color="darkorange")
    min_l = np.argmin(mean_c)
    ax.scatter([min_l], [mean_c[min_l]], color="red", s=100, zorder=5)
    ax.annotate(f"Min: L{min_l} ({mean_c[min_l]:.4f})", (min_l, mean_c[min_l]),
                textcoords="offset points", xytext=(10,-15), fontsize=11, color="red", fontweight="bold")
    ax.set_xlabel("Layer", fontsize=12); ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title(f"{model_short}: Forward vs Reverse Hidden Similarity ({n} pairs)", fontsize=13)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    fig.savefig(save_dir / f"cos_sim_{model_short}{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: cos_sim_{model_short}{tag}.png")

    output = {
        "model": model_short, "n_pairs": n, "n_layers": n_layers,
        "n_frames": args.n_frames,
        "tau_norm_peak": int(peak), "rel_tau_peak": int(peak_r),
        "rel_tau_peak_val": float(mean_r[peak_r]),
        "cos_sim_min": int(min_l),
        "rel_tau_mean_per_layer": mean_r.tolist(),
        "tau_norm_mean_per_layer": mean.tolist(),
    }
    with open(save_dir / f"tau_norm_{model_short}{tag}.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{model_short}: rel-τ peak=L{peak_r} ({mean_r[peak_r]:.4f}), "
          f"||τ|| peak=L{peak}, cos-sim min=L{min_l}")
    print(f"  TAI src layer: L{peak_r} (auto-selected via --tau_profile; injection starts at L{peak_r + 1})")


if __name__ == "__main__":
    main()
