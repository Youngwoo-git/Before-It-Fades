"""
tai_core.py — Temporal Activation Injection (TAI)
==================================================
Training-free temporal steering for video LLMs. TAI extracts a per-input
temporal steering vector τ = h(V) − h(V_rev) — the difference between
last-token hidden states of the forward and the temporally-reversed video —
at a source layer, and re-injects it into subsequent layers within a single
forward pass.

Model families are auto-detected from the model name:
qwen (Qwen2.5-VL / Qwen3-VL), internvl, molmo2, gemma4.

Key arguments (shared by the evaluation scripts):
  --method        baseline | tai
  --tau_profile   τ norm JSON from tau_profile/tau_norm_multi_model.py.
                  Enables profile-guided injection weights; the source layer
                  defaults to the profile peak. Without a profile, a
                  linear-decay schedule from --src_layer is used.
  --src_layer     layer where τ is extracted (default: the τ profile peak;
                  without a profile it must be set explicitly)
  --inject_start  first layer receiving injection (default: src_layer + 1)
  --beta          injection strength; negative β steers away from time
  --n_frames      frames sampled per video

Implementation notes: the reverse pass exits early at src_layer, and extraction
+ injection happen in one pass on the original video. lm_head is patched to
score only the last token; all hooks are removed and VRAM freed after every call.
"""
import torch
import re, gc, os
import numpy as np

def _jitter_rng(sample_seed, video_path):
    import hashlib
    h = int(hashlib.md5(f"{sample_seed}:{video_path}".encode()).hexdigest()[:8], 16)
    return np.random.RandomState(h)

def _jitter_indices(total, n_frames, sample_seed, video_path):
    base = np.linspace(0, max(total - 1, 0), n_frames)
    if sample_seed is None:
        return base.astype(int)
    rng = _jitter_rng(sample_seed, video_path)
    gap = (max(total - 1, 1)) / max(n_frames - 1, 1)
    jit = rng.uniform(-gap / 2, gap / 2, size=n_frames)
    return np.sort(np.clip(np.round(base + jit), 0, max(total - 1, 0)).astype(int))

# ============================================================
# 1. Model Adapter
# ============================================================
class ModelAdapter:
    def __init__(self, model, processor, tokenizer, model_name, family):
        self.model = model; self.processor = processor; self.tokenizer = tokenizer
        self.model_name = model_name; self.family = family; self._orig_lm = None
    def get_layers(self):
        m = self.model
        for path in ["model.layers","model.language_model.layers","language_model.model.layers",
                     "model.transformer.blocks","transformer.blocks"]:
            obj = m
            try:
                for attr in path.split("."): obj = getattr(obj, attr)
                if hasattr(obj,'__len__') and len(obj) > 10: return obj
            except AttributeError: continue
        raise ValueError(f"Cannot find layers for {self.model_name}")
    def get_lm_head(self):
        m = self.model
        for path in ["lm_head","language_model.lm_head","language_model.output",
                     "model.lm_head","model.transformer.ff_out"]:
            obj = m
            try:
                for attr in path.split("."): obj = getattr(obj, attr)
                return obj
            except AttributeError: continue
        raise ValueError(f"Cannot find lm_head for {self.model_name}")
    def patch_lm_head(self):
        lm = self.get_lm_head(); self._orig_lm = lm.forward
        lm.forward = lambda x: self._orig_lm(x[:, -1:, :])
    def unpatch_lm_head(self):
        if self._orig_lm is not None:
            lm = self.get_lm_head(); lm.forward = self._orig_lm; self._orig_lm = None
    def n_layers(self): return len(self.get_layers())
    def device(self): return next(self.model.parameters()).device
    def forward_model(self, inputs):
        """Unified forward: handles InternVL inputs_embeds case."""
        if self.family == "internvl" and "inputs_embeds" in inputs:
            return self.model.language_model(**inputs)
        return self.model(**inputs)

def detect_family(mn):
    mn = mn.lower()
    if "molmo" in mn: return "molmo2"
    if "gemma" in mn: return "gemma4"
    if "qwen" in mn: return "qwen"
    if "internvl" in mn: return "internvl"
    raise ValueError(f"Unknown model family: {mn} "
                     f"(supported: qwen, internvl, molmo2, gemma4)")

def load_model_adapter(model_name):
    family = detect_family(model_name)
    print(f"Loading {model_name} (family: {family})...")
    if family == "qwen":
        if "Qwen2.5" in model_name or "Qwen2_5" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
            base = model_name if (model_name.startswith("Qwen/") or
                                  os.path.isfile(os.path.join(model_name, "preprocessor_config.json"))
                                  ) else "Qwen/Qwen2.5-VL-7B-Instruct"
        else:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
            base = model_name if model_name.startswith("Qwen/") else "Qwen/Qwen3-VL-8B-Instruct"
        model.eval()
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base)
        a = ModelAdapter(model, processor, processor.tokenizer, model_name, "qwen")
    elif family == "molmo2":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        a = ModelAdapter(model, processor, processor.tokenizer, model_name, "molmo2")
    elif family == "gemma4":
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForMultimodalLM as _GemmaAM
        except ImportError:
            from transformers import AutoModelForImageTextToText as _GemmaAM
        model = _GemmaAM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
            device_map="auto", low_cpu_mem_usage=True)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)
        print(f"  [gemma4] loaded via {_GemmaAM.__name__}")
        a = ModelAdapter(model, processor, processor.tokenizer, model_name, "gemma4")
    elif family == "internvl":
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        model.eval(); tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        a = ModelAdapter(model, None, tokenizer, model_name, "internvl")
    print(f"  {a.n_layers()} layers"); return a

# ============================================================
# 2. Video Input Preparation (n_frames parameterized)
# ============================================================
def sanitize_video_kwargs(vk):
    if "fps" in vk and isinstance(vk["fps"], (list, tuple)): vk["fps"] = float(vk["fps"][0])
    return vk

def prepare_input(adapter, video_path, prompt, n_frames=16, sample_seed=None):
    """Prepare model inputs. Returns (inputs, videos, video_kwargs)."""
    if adapter.family == "qwen":
        from qwen_vl_utils import process_vision_info
        patch_size = 16 if "Qwen3" in adapter.model_name else 14
        video_item = {"type":"video","video":video_path,"max_pixels":720*720,"nframes":n_frames}
        if sample_seed is not None:
            import av as _av
            _c = _av.open(video_path)
            _dur = float(_c.duration / 1e6) if _c.duration else None
            _c.close()
            if not _dur or _dur <= 0:
                raise RuntimeError(f"jitter: duration probe failed for {video_path}")
            rng = _jitter_rng(sample_seed, video_path)
            video_item["video_start"] = float(rng.uniform(0, _dur / max(n_frames, 1)))
        messages = [{"role":"user","content":[video_item, {"type":"text","text":prompt}]}]
        try:
            text = adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, vk = process_vision_info(messages, image_patch_size=patch_size, return_video_kwargs=True)
        except ValueError as ve:
            match = re.search(r"interval \[\d+, (\d+)\]", str(ve))
            if match:
                messages[0]["content"][0]["nframes"] = int(match.group(1))
                text = adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos, vk = process_vision_info(messages, image_patch_size=patch_size, return_video_kwargs=True)
            else: raise
        vk = sanitize_video_kwargs(vk)
        inputs = adapter.processor(text=[text], images=images, videos=videos, return_tensors="pt", **vk)
        inputs = {k: v.to(adapter.device()) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        return inputs, videos, vk
    elif adapter.family == "molmo2":
        import av
        container = av.open(video_path)
        total = container.streams.video[0].frames
        if total <= 0: total = sum(1 for _ in container.decode(video=0)); container.seek(0)
        indices = _jitter_indices(total, n_frames, sample_seed, video_path)
        idx_set = set(indices.tolist())
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in idx_set: frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames: break
        container.close()
        clip = np.stack(frames) if frames else np.zeros((n_frames,224,224,3), dtype=np.uint8)
        messages = [{"role":"user","content":[
            {"type":"video","video":video_path},
            {"type":"text","text":prompt}]}]
        text = adapter.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        try:
            from transformers.video_utils import VideoMetadata
        except ImportError:
            from transformers.utils.video_utils import VideoMetadata
        meta = VideoMetadata(total_num_frames=len(clip), fps=2.0,
                             duration=len(clip) / 2.0, video_backend="numpy")
        inputs = adapter.processor(text=[text], videos=[clip],
                                   video_metadata=[meta], return_tensors="pt")
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(adapter.device())
                if v.dtype == torch.float32 and ("pixel" in k or "video" in k or "image" in k):
                    v = v.to(torch.bfloat16)
            out[k] = v
        return out, [clip], {}
    elif adapter.family == "gemma4":
        import av
        container = av.open(video_path)
        total = container.streams.video[0].frames
        if total <= 0: total = sum(1 for _ in container.decode(video=0)); container.seek(0)
        indices = _jitter_indices(total, n_frames, sample_seed, video_path)
        idx_set = set(indices.tolist())
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in idx_set: frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames: break
        container.close()
        clip = np.stack(frames) if frames else np.zeros((n_frames,224,224,3), dtype=np.uint8)
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in clip]
        content = [{"type": "image", "image": fr} for fr in pil_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = adapter.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(adapter.device())
                if v.dtype == torch.float32 and ("pixel" in k or "video" in k or "image" in k):
                    v = v.to(torch.bfloat16)
            out[k] = v
        return out, [clip], {}
    elif adapter.family == "internvl":
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        from PIL import Image
        transform = T.Compose([T.Lambda(lambda img: img.convert('RGB') if hasattr(img,'convert') else img),
            T.Resize((448,448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
            T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))])
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(video_path, ctx=cpu(0)); total = len(vr)
            indices = np.linspace(0, total-1, n_frames).astype(int)
            frames = [Image.fromarray(vr[i].asnumpy()) for i in indices]
        except Exception:
            import cv2; cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, max(total-1,0), n_frames).astype(int); frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ret, fr = cap.read()
                if ret: frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            cap.release()
        pv = torch.stack([transform(f) for f in frames]).to(adapter.device(), dtype=torch.bfloat16)
        img_ctx = '<IMG_CONTEXT>' * 256
        vp_str = ''.join([f'Frame{i+1}: {img_ctx}\n' for i in range(len(frames))])
        tokenizer = adapter.tokenizer
        img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        template = f'<|im_start|>user\n{vp_str + prompt}<|im_end|>\n<|im_start|>assistant\n'
        input_ids = tokenizer(template, return_tensors='pt').input_ids.to(adapter.device())
        with torch.no_grad(): vit_embeds = adapter.model.extract_feature(pv)
        input_embeds = adapter.model.language_model.get_input_embeddings()(input_ids)
        img_mask = (input_ids == img_ctx_id)
        if img_mask.sum() > 0 and vit_embeds is not None:
            vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
            n_img = img_mask.sum().item()
            if n_img <= vit_flat.shape[0]:
                input_embeds[img_mask] = vit_flat[:n_img].to(input_embeds.device, dtype=input_embeds.dtype)
        inputs = {"inputs_embeds": input_embeds, "attention_mask": torch.ones_like(input_ids).to(input_embeds.device)}
        return inputs, [pv], {}

def prepare_reverse_input(adapter, videos, prompt, video_kwargs=None, n_frames=16):
    """Create reverse video input. Only needs videos tensor + prompt (not original inputs)."""
    if video_kwargs is None: video_kwargs = {}
    if videos is None: return None
    if adapter.family == "qwen":
        videos_rev = [torch.flip(v, dims=[0]) if isinstance(v, torch.Tensor) else v for v in videos]
        messages = [{"role":"user","content":[
            {"type":"video","video":"dummy","max_pixels":720*720,"nframes":n_frames},
            {"type":"text","text":prompt}]}]
        text_t = adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs_rev = adapter.processor(text=[text_t], images=None, videos=videos_rev, return_tensors="pt", **video_kwargs)
        return {k: v.to(adapter.device()) if isinstance(v, torch.Tensor) else v for k, v in inputs_rev.items()}
    elif adapter.family == "molmo2":
        clip_rev = videos[0][::-1].copy()
        messages = [{"role":"user","content":[
            {"type":"video","video":"dummy"},
            {"type":"text","text":prompt}]}]
        text_t = adapter.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        try:
            from transformers.video_utils import VideoMetadata
        except ImportError:
            from transformers.utils.video_utils import VideoMetadata
        meta = VideoMetadata(total_num_frames=len(clip_rev), fps=2.0,
                             duration=len(clip_rev) / 2.0, video_backend="numpy")
        inputs_rev = adapter.processor(text=[text_t], videos=[clip_rev],
                                       video_metadata=[meta], return_tensors="pt")
        out = {}
        for k, v in inputs_rev.items():
            if isinstance(v, torch.Tensor):
                v = v.to(adapter.device())
                if v.dtype == torch.float32 and ("pixel" in k or "video" in k or "image" in k):
                    v = v.to(torch.bfloat16)
            out[k] = v
        return out
    elif adapter.family == "gemma4":
        clip_rev = videos[0][::-1].copy()
        from PIL import Image
        pil_rev = [Image.fromarray(f) for f in clip_rev]
        content = [{"type": "image", "image": fr} for fr in pil_rev]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs_rev = adapter.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        out = {}
        for k, v in inputs_rev.items():
            if isinstance(v, torch.Tensor):
                v = v.to(adapter.device())
                if v.dtype == torch.float32 and ("pixel" in k or "video" in k or "image" in k):
                    v = v.to(torch.bfloat16)
            out[k] = v
        return out
    elif adapter.family == "internvl":
        pv_rev = videos[0].flip(dims=[0])
        img_ctx = '<IMG_CONTEXT>' * 256
        vp_str = ''.join([f'Frame{i+1}: {img_ctx}\n' for i in range(pv_rev.shape[0])])
        tokenizer = adapter.tokenizer
        img_ctx_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        template = f'<|im_start|>user\n{vp_str + prompt}<|im_end|>\n<|im_start|>assistant\n'
        input_ids = tokenizer(template, return_tensors='pt').input_ids.to(adapter.device())
        with torch.no_grad(): vit_embeds = adapter.model.extract_feature(pv_rev)
        input_embeds = adapter.model.language_model.get_input_embeddings()(input_ids)
        img_mask = (input_ids == img_ctx_id)
        if img_mask.sum() > 0 and vit_embeds is not None:
            vit_flat = vit_embeds.reshape(-1, vit_embeds.shape[-1])
            n_img = img_mask.sum().item()
            if n_img <= vit_flat.shape[0]:
                input_embeds[img_mask] = vit_flat[:n_img].to(input_embeds.device, dtype=input_embeds.dtype)
        return {"inputs_embeds": input_embeds, "attention_mask": torch.ones_like(input_ids).to(input_embeds.device)}

# ============================================================
# 3. TAI Core
# ============================================================
class EarlyExit(Exception): pass

@torch.no_grad()
def extract_hidden(adapter, inputs, src_layer):
    """EarlyExit extraction: forward only up to src_layer, then abort."""
    captured = {}; layers = adapter.get_layers()
    def hook(m, i, o):
        out = o[0] if isinstance(o, tuple) else o
        captured['h'] = out[0, -1, :].detach().clone(); raise EarlyExit()
    handle = layers[src_layer].register_forward_hook(hook)
    adapter.patch_lm_head()
    try: adapter.forward_model(inputs)
    except EarlyExit: pass
    finally: adapter.unpatch_lm_head(); handle.remove()
    return captured['h']

@torch.no_grad()
def extract_hidden_window(adapter, inputs, src_layer, window):
    """Extract averaged hidden from [src-w, ..., src+w]. EarlyExit at last."""
    w = window // 2
    n = adapter.n_layers(); layers = adapter.get_layers()
    targets = list(range(max(0, src_layer - w), min(n, src_layer + w + 1)))
    last = max(targets); captured = {}; handles = []
    for li in targets:
        def make_hook(idx):
            def hook_fn(m, i, o):
                out = o[0] if isinstance(o, tuple) else o
                captured[idx] = out[0, -1, :].detach().clone()
                if idx == last: raise EarlyExit()
            return hook_fn
        handles.append(layers[li].register_forward_hook(make_hook(li)))
    adapter.patch_lm_head()
    try: adapter.forward_model(inputs)
    except EarlyExit: pass
    finally:
        adapter.unpatch_lm_head()
        for h in handles: h.remove()
    dev = captured[last].device
    return torch.stack([captured[li].to(dev) for li in targets]).mean(dim=0)

@torch.no_grad()
def tai_forward(adapter, inputs, inputs_rev, src_layer, inject_start, beta,
                tau_profile=None, src_window=1):
    """
    TAI forward: extraction + injection in a single pass on the original video.
    src_window: if >1, average h from [src-w, ..., src+w] before computing τ.
    """
    layers = adapter.get_layers()
    n = len(layers)

    # Window setup
    w = src_window // 2
    window_layers = list(range(max(0, src_layer - w), min(n, src_layer + w + 1)))
    last_wl = max(window_layers)

    # ############################### TAI (1/3) ###############################
    # Reverse pass: h_rev = last-token hidden state of the REVERSED video,
    # captured at src_layer. EarlyExit aborts the forward right after src_layer.
    # #########################################################################
    if src_window > 1:
        h_rev = extract_hidden_window(adapter, inputs_rev, src_layer, src_window)
    else:
        h_rev = extract_hidden(adapter, inputs_rev, src_layer)
    del inputs_rev; torch.cuda.empty_cache()

    # ############################### TAI (2/3) ###############################
    # Injection schedule: for every layer l >= inject_start, add β·w_l·τ to the
    # LAST-TOKEN residual stream (o2[0, -1, :] += ...).
    #   - guided:  w_l = normalized τ profile weight for layer l (from JSON)
    #   - linear:  w_l decays linearly from 1 at inject_start to 0 at the top
    # Hooks only fire on the prefill pass (o2.shape[1] > 1 guard).
    # #########################################################################
    inject_handles = []
    def register_inject(tau_vec, eff_inj_start):
        if tau_profile is not None:
            for li in range(eff_inj_start, n):
                wt = tau_profile[li] if li < len(tau_profile) else 0.0
                if wt < 1e-6: continue
                b = beta * wt
                def make_inject(b_val, tv):
                    def hook_fn(mod, inp2, out2):
                        o2 = out2[0] if isinstance(out2, tuple) else out2
                        if o2.shape[1] > 1:
                            # <<< TAI injection: h_l[last] += β·w_l·τ >>>
                            o2[0, -1, :] += b_val * tv.to(o2.device, dtype=o2.dtype)
                        return out2 if not isinstance(out2, tuple) else (o2,) + out2[1:]
                    return hook_fn
                inject_handles.append(layers[li].register_forward_hook(make_inject(b, tau_vec)))
        else:
            for li in range(eff_inj_start, n):
                b = beta * (1.0 - ((li - eff_inj_start) / max(1, n - eff_inj_start - 1)))
                def make_inject(b_val, tv):
                    def hook_fn(mod, inp2, out2):
                        o2 = out2[0] if isinstance(out2, tuple) else out2
                        if o2.shape[1] > 1:
                            # <<< TAI injection: h_l[last] += β·w_l·τ >>>
                            o2[0, -1, :] += b_val * tv.to(o2.device, dtype=o2.dtype)
                        return out2 if not isinstance(out2, tuple) else (o2,) + out2[1:]
                    return hook_fn
                inject_handles.append(layers[li].register_forward_hook(make_inject(b, tau_vec)))

    # ############################### TAI (3/3) ###############################
    # Single forward pass on the ORIGINAL video with dynamic hook registration:
    # when the forward reaches src_layer, capture h_fwd (last token), compute
    #     τ = h_fwd − h_rev
    # on the fly, and immediately register the injection hooks of (2/3) for all
    # downstream layers — so extraction + injection happen in ONE forward.
    # #########################################################################
    capture = {}
    all_hooks = []

    if src_window <= 1:
        # Original single-layer path
        def src_hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            h_orig = o[0, -1, :].detach().clone()
            tau_vec = h_orig - h_rev.to(h_orig.device)   # τ = h(V) − h(V_rev)
            capture['tau'] = tau_vec
            inj = inject_start if inject_start > src_layer else src_layer + 1
            register_inject(tau_vec, inj)

        all_hooks.append(layers[src_layer].register_forward_hook(src_hook))
    else:
        # Window path: hook all window layers, compute tau on last
        fwd_captured = {}
        for wl in window_layers:
            def make_whook(layer_idx):
                def hook_fn(module, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    fwd_captured[layer_idx] = o[0, -1, :].detach().clone()
                    if layer_idx == last_wl:
                        dev = fwd_captured[last_wl].device
                        h_avg = torch.stack([fwd_captured[li].to(dev) for li in window_layers]).mean(dim=0)
                        tau_vec = h_avg - h_rev.to(h_avg.device)
                        capture['tau'] = tau_vec
                        inj = inject_start if inject_start > last_wl else last_wl + 1
                        register_inject(tau_vec, inj)
                return hook_fn
            all_hooks.append(layers[wl].register_forward_hook(make_whook(wl)))

    adapter.patch_lm_head()
    try:
        outputs = adapter.forward_model(inputs)
        logits = outputs.logits[0, -1, :].clone()
        del outputs
    finally:
        adapter.unpatch_lm_head()
        for h in all_hooks: h.remove()
        for h in inject_handles: h.remove()
        del h_rev; torch.cuda.empty_cache()
    return logits

@torch.no_grad()
def baseline_forward(adapter, inputs):
    """Simple baseline forward. Returns logits."""
    adapter.patch_lm_head()
    try:
        outputs = adapter.forward_model(inputs)
        logits = outputs.logits[0, -1, :].clone()
        del outputs
    finally:
        adapter.unpatch_lm_head()
    return logits

@torch.no_grad()
def run_inference(adapter, video_path, prompt, method, src_layer, inject_start, beta, n_frames,
                  tau_profile=None, src_window=1, sample_seed=None):
    """
    Unified inference entry point. Methods: baseline, tai.
    """
    inputs, videos, vk = prepare_input(adapter, video_path, prompt, n_frames=n_frames, sample_seed=sample_seed)

    if method == "tai" and videos is not None:
        inputs_rev = prepare_reverse_input(adapter, videos, prompt, vk, n_frames=n_frames)
        if inputs_rev is not None:
            logits = tai_forward(adapter, inputs, inputs_rev, src_layer, inject_start, beta,
                                 tau_profile=tau_profile, src_window=src_window)
        else:
            logits = baseline_forward(adapter, inputs)
    else:
        logits = baseline_forward(adapter, inputs)

    torch.cuda.empty_cache()
    return logits

# ============================================================
# 4. τ Profile Utilities
# ============================================================
import json as _json

def load_tau_profile(json_path, clip_last=0, inverse=False):
    """Load τ norm profile from JSON and compute per-layer injection weights.
    Args:
        clip_last: int, set last N layers to 0 weight (suppress last-layer spike).
        inverse: bool, if True use weight = (1 - normalized_tau) instead of normalized_tau.
                 Inject MORE where τ is SMALL (sanity check / counter-hypothesis).
                 Returned src_layer is argmin (smallest τ) instead of argmax.
    Returns:
        tau_profile: list of floats [n_layers], where peak weight=1.0 and others proportional.
        src_layer: int, the recommended source layer (argmax for normal, argmin for inverse).
    """
    with open(json_path) as f:
        data = _json.load(f)
    rel_tau = data["rel_tau_mean_per_layer"]
    peak_val = max(rel_tau)
    norm_tau = [v / (peak_val + 1e-8) for v in rel_tau]
    if inverse:
        # Weight = 1 - normalized_tau, then re-normalize so max=1.0
        inv = [1.0 - v for v in norm_tau]
        inv_max = max(inv) + 1e-8
        tau_profile = [v / inv_max for v in inv]
        # src_layer = argmin of original τ (where injection weight is highest in inverse mode)
        src_recommended = rel_tau.index(min(rel_tau))
        print(f"  τ profile (inverse): src=L{src_recommended}, n_layers={len(tau_profile)}")
    else:
        tau_profile = norm_tau
        src_recommended = rel_tau.index(peak_val)
        print(f"  τ profile: peak=L{src_recommended}, n_layers={len(tau_profile)}")
    if clip_last > 0:
        for i in range(max(0, len(tau_profile) - clip_last), len(tau_profile)):
            tau_profile[i] = 0.0
    return tau_profile, src_recommended

# ============================================================
# 5. Classification Utilities
# ============================================================
def get_token_ids(tokenizer, choices="ABCDE"):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)[0]
    return {c: encode(c) for c in choices}

def classify_from_logits(logits, token_ids, valid_choices):
    scores = {c: logits[token_ids[c]].item() for c in valid_choices if c in token_ids}
    return max(scores, key=scores.get) if scores else valid_choices[0]
