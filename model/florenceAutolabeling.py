import os
import glob
import json
import re
import argparse
from pathlib import Path
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================

DEFAULT_FLORENCE2_MODEL_ID = os.environ.get(
    "FLORENCE2_MODEL_ID",
    "mlx-community/Florence-2-large-ft-bf16"   # HuggingFace repo slug
)
# Or point to a local folder:  ~/models/Florence-2-large-ft-bf16
DEFAULT_FLORENCE2_LOCAL_PATH = os.environ.get("FLORENCE2_LOCAL_PATH", None)

OUTPUT_DIR = "./outputs"
PET_KEYWORDS = ["cat", "dog"]
RISK_OBJECTS = ["cup", "laptop", "potted plant", "vase", "remote", "keyboard"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# MODEL LOADING  (mlx-vlm / local mlx)
# ==========================================

florence2_model = None
florence2_processor = None
florence2_device = None


def _apply_florence_compatibility_patches():
    try:
        import torch
        import torch.nn as nn
        from transformers.configuration_utils import PretrainedConfig
        from transformers.modeling_utils import PreTrainedModel
    except ImportError:
        return

    PretrainedConfig.forced_bos_token_id = None
    original_pretrained_init = PretrainedConfig.__init__

    def _patched_pretrained_init(self, *args, **kwargs):
        if "forced_bos_token_id" in kwargs:
            object.__setattr__(self, "forced_bos_token_id", kwargs["forced_bos_token_id"])
        return original_pretrained_init(self, *args, **kwargs)

    PretrainedConfig.__init__ = _patched_pretrained_init

    if not hasattr(PreTrainedModel, "_supports_sdpa"):
        PreTrainedModel._supports_sdpa = False
    if not hasattr(PreTrainedModel, "_supports_xformers"):
        PreTrainedModel._supports_xformers = False

    original_nn_getattr = nn.Module.__getattr__

    def _patched_nn_getattr(self, name):
        if name in {"_supports_sdpa", "_supports_xformers"}:
            return False
        return original_nn_getattr(self, name)

    nn.Module.__getattr__ = _patched_nn_getattr


def load_florence2(model_path: str):
    """
    Load Florence-2 directly via mlx_vlm (Apple Silicon, no HTTP server needed).
    Falls back to transformers+torch if mlx_vlm is unavailable.
    """
    global florence2_model, florence2_processor, _backend

    _apply_florence_compatibility_patches()

    # ── try mlx-vlm first (native Apple Silicon path) ──────────────────────
    try:
        from mlx_vlm import load as mlx_vlm_load
        print(f"Loading Florence-2 via mlx-vlm from: {model_path}")
        florence2_model, florence2_processor = mlx_vlm_load(model_path)
        _backend = "mlx"
        print("✅  Florence-2 loaded via mlx-vlm.")
        return "mlx"
    except ImportError:
        print("mlx-vlm not found, falling back to transformers …")
    except Exception as e:
        print(f"mlx-vlm load failed ({e}), falling back to transformers …")

    # ── transformers fallback (still runs on MPS on M1) ────────────────────
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
        from transformers.configuration_utils import PretrainedConfig
        from transformers.modeling_utils import PreTrainedModel

        # Compatibility patch for Florence-2 config objects and attention dispatch.
        PretrainedConfig.forced_bos_token_id = None
        original_pretrained_init = PretrainedConfig.__init__

        def _patched_pretrained_init(self, *args, **kwargs):
            if "forced_bos_token_id" in kwargs:
                object.__setattr__(self, "forced_bos_token_id", kwargs["forced_bos_token_id"])
            return original_pretrained_init(self, *args, **kwargs)

        PretrainedConfig.__init__ = _patched_pretrained_init

        if not hasattr(PreTrainedModel, "_supports_sdpa"):
            PreTrainedModel._supports_sdpa = False
        if not hasattr(PreTrainedModel, "_supports_xformers"):
            PreTrainedModel._supports_xformers = False

        _orig_module_getattr = nn.Module.__getattr__

        def _patched_module_getattr(self, name):
            if name in {"_supports_sdpa", "_supports_xformers"}:
                return False
            return _orig_module_getattr(self, name)

        nn.Module.__getattr__ = _patched_module_getattr

        device = (
            "mps" if (
                getattr(__import__("torch").backends, "mps", None) is not None
                and __import__("torch").backends.mps.is_available()
            ) else "cpu"
        )
        dtype = __import__("torch").float16

        print(f"Loading Florence-2 via transformers on {device} …")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if getattr(config.vision_config, "model_type", "") == "":
            print("Warning: Florence-2 vision_config.model_type missing, forcing to 'davit'.")
            config.vision_config.model_type = "davit"

        florence2_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            config=config,
        ).eval().to(device)
        florence2_processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        _backend = "transformers"
        globals()["florence2_device"] = device
        print(f"✅  Florence-2 loaded via transformers on {device}.")
        return "transformers"
    except Exception as e:
        raise RuntimeError(f"Could not load Florence-2 from '{model_path}': {e}") from e


# ==========================================
# INFERENCE
# ==========================================

def run_florence2_mlx(image: Image.Image, prompt: str, max_tokens: int = 256) -> str:
    from mlx_vlm import generate as mlx_vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    # mlx-vlm's Florence-2 adapter accepts the task token directly
    result = mlx_vlm_generate(
        florence2_model,
        florence2_processor,
        image=image,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    # result is a string
    return result.strip() if isinstance(result, str) else str(result).strip()


def run_florence2_transformers(image: Image.Image, prompt: str, max_tokens: int = 256) -> str:
    import torch

    if hasattr(florence2_model, "device"):
        device = florence2_model.device
    elif globals().get("florence2_device") is not None:
        device = globals()["florence2_device"]
    else:
        try:
            params = florence2_model.parameters()
            if isinstance(params, dict):
                params = params.values()
            first_param = next(iter(params))
            device = first_param.device
        except Exception:
            device = torch.device("cpu")

    inputs = florence2_processor(text=prompt, images=image, return_tensors="pt")
    inputs = {
        k: v.to(device, dtype=torch.float16) if k == "pixel_values" else v.to(device)
        for k, v in inputs.items()
    }
    with torch.no_grad():
        generated_ids = florence2_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tokens,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
    return florence2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


_backend = None  # set after load_florence2()


def run_florence2(image_path: str, prompt: str = "<MORE_DETAILED_CAPTION>",
                  max_tokens: int = 256) -> str:
    image = Image.open(image_path).convert("RGB")
    if _backend == "mlx":
        return run_florence2_mlx(image, prompt, max_tokens)
    else:
        return run_florence2_transformers(image, prompt, max_tokens)


# ==========================================
# RISK PARSING
# ==========================================

def parse_risk_output(generated_text: str) -> tuple[str, str]:
    text = generated_text.strip()
    if not text:
        return "LOW", "empty model response"

    lower_text = text.lower()
    found_pets = [p for p in PET_KEYWORDS if re.search(rf'\b{p}\b', lower_text)]
    found_objs = [o for o in RISK_OBJECTS if re.search(rf'\b{o}\b', lower_text)]

    if not found_pets or not found_objs:
        return "LOW", "no pet+object pair detected"

    high_pattern = (
        r'\b(approaching|touching|manipulating|grabbing|holding|pawing|pressing|'
        r'hitting|jumping on|climbing|chewing|eating|drinking|reaching for|using|'
        r'playing with)\b'
    )
    medium_pattern = (
        r'\b(near|close to|next to|beside|nearby|watching|looking at|'
        r'by the|beside the)\b'
    )

    m = re.search(high_pattern, lower_text)
    if m:
        return "HIGH", f"clear imminent interaction: {m.group(1)}"

    m = re.search(medium_pattern, lower_text)
    if m:
        return "MEDIUM", "pet near object, possible interaction"

    return "LOW", "pet and object present but no clear action"


# ==========================================
# LABELING
# ==========================================

def label_pet_risk_for_image(image_path: str, max_tokens: int = 256) -> dict:
    try:
        generated_text = run_florence2(image_path, max_tokens=max_tokens)
    except Exception as e:
        print(f"  ⚠️  Inference failed for {os.path.basename(image_path)}: {e}")
        generated_text = ""

    risk_level, reason = parse_risk_output(generated_text)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    return {
        "clip_id": image_id,
        "frames": [image_id],
        "risk_level": risk_level,
        "reason": reason,
        "raw": generated_text,
    }


def label_pet_risk_directory(image_dir: str, output_dir: str, max_tokens: int = 256):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(sorted(glob.glob(os.path.join(image_dir, ext))))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "INVALID": 0}

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {os.path.basename(image_path)}")
        result = label_pet_risk_for_image(image_path, max_tokens)
        output_path = os.path.join(output_dir, f"{result['clip_id']}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        level = result["risk_level"]
        counts[level if level in counts else "INVALID"] += 1
        print(f"  → {level}: {result['reason']}")
        print(f"  Raw: {result['raw'][:120]}{'…' if len(result['raw']) > 120 else ''}")

    print("\n── Summary ──────────────────")
    for k in ("HIGH", "MEDIUM", "LOW", "INVALID"):
        if counts[k]:
            print(f"  {k}: {counts[k]}")


# ==========================================
# MAIN
# ==========================================

def main():
    global _backend

    parser = argparse.ArgumentParser(
        description="Florence-2 pet risk labeler — runs locally via mlx-vlm on Apple Silicon"
    )
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to a single image file")
    parser.add_argument("--image_dir", type=str, default="./dataset/test/images",
                        help="Directory of images to label")
    parser.add_argument("--label_dir", type=str, default="./dataset/test/labels",
                        help="Output directory for JSON label files")
    parser.add_argument("--model_path", type=str,
                        default=DEFAULT_FLORENCE2_LOCAL_PATH or DEFAULT_FLORENCE2_MODEL_ID,
                        help=(
                            "Local folder OR HuggingFace repo slug for Florence-2. "
                            "Examples:\n"
                            "  ~/models/Florence-2-large-ft-bf16\n"
                            "  mlx-community/Florence-2-large-ft-bf16"
                        ))
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max new tokens for the vision model")
    args = parser.parse_args()

    _backend = load_florence2(args.model_path)

    os.makedirs(args.label_dir, exist_ok=True)

    if args.image_path and os.path.isfile(args.image_path):
        result = label_pet_risk_for_image(args.image_path, args.max_tokens)
        output_path = os.path.join(args.label_dir, f"{result['clip_id']}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved : {output_path}")
        print(f"Risk  : {result['risk_level']} — {result['reason']}")
        print(f"Raw   : {result['raw']}")
    else:
        label_pet_risk_directory(args.image_dir, args.label_dir, args.max_tokens)


if __name__ == "__main__":
    main()