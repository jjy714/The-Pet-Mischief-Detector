import argparse
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

try:
    from groundingdino.util.inference import load_image as load_gd_image
    from groundingdino.util.inference import load_model as load_gd_model
    from groundingdino.util.inference import predict as gd_predict
except ImportError as exc:
    raise ImportError(
        "GroundingDINO is required. Install it from GitHub. "
        "Run:\n  pip install git+https://github.com/IDEA-Research/GroundingDINO.git"
    ) from exc

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:
    raise ImportError(
        "Segment Anything is required. Install it from GitHub. "
        "Run:\n  pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from exc

PET_KEYWORDS = ["cat", "dog"]
RISK_OBJECTS = ["cup", "laptop", "potted plant", "vase", "remote", "keyboard"]
OBJECT_PROMPT = ", ".join(PET_KEYWORDS + RISK_OBJECTS)

DEFAULT_OMLX_ENDPOINT = os.environ.get("OMLX_ENDPOINT", "http://127.0.0.1:8000")
DEFAULT_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen2.5-VL-7B-Instruct-8bit")
DEFAULT_GD_CONFIG = "./configs/GroundingDINO_SwinB.cfg"
DEFAULT_GD_CHECKPOINT = "./weights/groundingdino_swint_ogc.pth"
DEFAULT_SAM_CHECKPOINT = "./weights/sam_vit_h_4b8939.pth"

VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}


def call_omlx(endpoint: str, model_id: str, prompt: str, max_tokens: int = 50) -> str:
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def test_connection(endpoint: str, model_id: str) -> bool:
    print(f"\n--- oMLX Connection Test ---")
    print(f"Endpoint : {endpoint}")
    print(f"Model ID : {model_id}")

    for path in ["/v1/models", "/health", "/"]:
        url = endpoint.rstrip("/") + path
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read(300).decode("utf-8", errors="replace")
                print(f"✅  HTTP {resp.status} at {url}")
                try:
                    models = json.loads(body).get("data", [])
                    if models:
                        print(f"   Available models: {[m['id'] for m in models]}")
                except Exception:
                    pass
            break
        except urllib.error.HTTPError as e:
            print(f"⚠️  HTTP {e.code} at {url}")
            break
        except urllib.error.URLError as e:
            print(f"❌  Cannot reach {url}: {e.reason}")
            print("\nIs oMLX running? Start it with:\n  omlx serve --model-dir ~/models")
            return False

    print(f"Sending smoke-test prompt to model '{model_id}' …")
    try:
        result = call_omlx(endpoint, model_id, "Reply with exactly one word: HIGH", max_tokens=5)
        print(f"✅  Model responded: '{result}'")
    except Exception as e:
        print(f"❌  Request failed: {e}")
        return False

    print("\n--- Test complete ---\n")
    return True


def load_groundingdino(config_path: Path, checkpoint_path: Path, device: str):
    if not config_path.exists():
        raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")

    model = load_gd_model(str(config_path), str(checkpoint_path), device)
    model.to(device)
    model.eval()
    return model


def load_sam(checkpoint_path: Path, device: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    return SamPredictor(sam)


def encode_mask_rle(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def normalize_label(phrase: str) -> str | None:
    text = phrase.strip().lower()
    for label in PET_KEYWORDS + RISK_OBJECTS:
        if label in text:
            return label
    return None


def build_vlm_prompt(clip_id: str, objects: list[dict]) -> str:
    lines = [
        f"Scene ID: {clip_id}",
        "You are a visual risk assessment model. Use the detected pet and risk objects to determine the overall risk.",
        "Only classify using these categories:\n"
        "  - Pets: cat, dog\n"
        "  - Risk objects: cup, laptop, potted plant, vase, remote, keyboard",
    ]

    if objects:
        lines.append("Detected objects:")
        for obj in objects:
            lines.append(
                f"- {obj['label']} (score={obj['score']:.2f}) bbox={obj['bbox']} mask_present={obj['mask_present']}"
            )
    else:
        lines.append("No pets or risk objects were detected in the frame.")

    lines.extend([
        "Classify the overall risk level as HIGH, MEDIUM, or LOW.",
        "Rules:",
        "  LOW: pet far from object or near but clearly inactive.",
        "  MEDIUM: pet near object, possible interaction but no clear action.",
        "  HIGH: clear imminent interaction (approaching fast, touching, manipulating object).",
        "  HIGH also when a cat or dog is lying, sitting, or resting on a keyboard, laptop, monitor, desk, or another risk object.",
        "Provide your answer as a JSON object with keys risk_level and reason only.",
        "Example output: {\"risk_level\":\"LOW\",\"reason\":\"The pet is not near any risk object.\"}",
    ])
    return "\n".join(lines)


def parse_vlm_response(response: str) -> tuple[str, str]:
    text = response.strip()
    try:
        parsed = json.loads(text)
        label = str(parsed.get("risk_level", "LOW")).strip().upper()
        if label not in VALID_LEVELS:
            label = "LOW"
        reason = str(parsed.get("reason", text)).strip()
        return label, reason
    except Exception:
        for token in re.split(r"[\s,{}\[\]\"]+", text.upper()):
            if token in VALID_LEVELS:
                return token, text
        return "LOW", text or "empty model response"


def detect_objects_with_groundingdino(image_path: Path, model, sam_predictor, device: str, box_threshold: float, text_threshold: float) -> list[dict]:
    image_pil = load_gd_image(str(image_path)).convert("RGB")
    image_np = np.array(image_pil)
    boxes, scores, phrases = gd_predict(
        model,
        image_pil,
        OBJECT_PROMPT,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    relevant_objects: list[dict] = []
    if len(boxes) == 0:
        return relevant_objects

    sam_predictor.set_image(image_np)
    for bbox, score, phrase in zip(boxes, scores, phrases):
        label = normalize_label(phrase)
        if label is None:
            continue
        box = np.array(bbox, dtype=float).reshape(1, 4)
        masks, _, _ = sam_predictor.predict(box=box, multimask_output=False)
        mask = masks[0] if len(masks) else np.zeros(image_np.shape[:2], dtype=bool)
        encoding = encode_mask_rle(mask)

        x0, y0, x1, y1 = [float(v) for v in bbox]
        relevant_objects.append({
            "label": label,
            "phrase": phrase,
            "score": float(score),
            "bbox": [x0, y0, x1, y1],
            "mask_present": bool(mask.any()),
            "mask_rle": encoding,
        })
    return relevant_objects


def build_output_record(clip_id: str, objects: list[dict], vlm_response: str) -> dict:
    risk_level, reason = parse_vlm_response(vlm_response)
    return {
        "clip_id": clip_id,
        "frames": [clip_id],
        "risk_level": risk_level,
        "reason": reason,
        "raw": vlm_response,
        "objects": objects,
    }


def process_image(image_path: Path, grounding_model, sam_predictor, endpoint: str, model_id: str, max_tokens: int, device: str, box_threshold: float, text_threshold: float) -> dict:
    clip_id = image_path.stem
    objects = detect_objects_with_groundingdino(
        image_path,
        grounding_model,
        sam_predictor,
        device,
        box_threshold,
        text_threshold,
    )
    prompt = build_vlm_prompt(clip_id, objects)
    vlm_response = call_omlx(endpoint, model_id, prompt, max_tokens=max_tokens)
    return build_output_record(clip_id, objects, vlm_response)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="GroundDINO + SAM + Qwen2.5-VL risk labeling pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory.")
    parser.add_argument("--output", type=str, default="./outputs/grounddino_sam_qwen2.5", help="Output directory for JSON results.")
    parser.add_argument("--omlx_endpoint", type=str, default=DEFAULT_OMLX_ENDPOINT, help="oMLX endpoint URL.")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Qwen model ID to use.")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens for the VLM response.")
    parser.add_argument("--grounding_config", type=str, default=DEFAULT_GD_CONFIG, help="GroundingDINO config file path.")
    parser.add_argument("--grounding_checkpoint", type=str, default=DEFAULT_GD_CHECKPOINT, help="GroundingDINO checkpoint path.")
    parser.add_argument("--sam_checkpoint", type=str, default=DEFAULT_SAM_CHECKPOINT, help="SAM checkpoint path.")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="GroundingDINO box confidence threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO text confidence threshold.")
    parser.add_argument("--test_connection", action="store_true", help="Test oMLX connection and exit.")
    args = parser.parse_args()

    if args.test_connection:
        ok = test_connection(args.omlx_endpoint, args.model_id)
        raise SystemExit(0 if ok else 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    grounding_model = load_groundingdino(Path(args.grounding_config), Path(args.grounding_checkpoint), device)
    sam_predictor = load_sam(Path(args.sam_checkpoint), device)

    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            [p for ext in ("*.jpg", "*.jpeg", "*.png") for p in input_path.glob(ext)]
        )
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        print(f"Processing {image_path.name}...")
        record = process_image(
            image_path,
            grounding_model,
            sam_predictor,
            args.omlx_endpoint,
            args.model_id,
            args.max_new_tokens,
            device,
            args.box_threshold,
            args.text_threshold,
        )
        save_path = out_dir / f"{image_path.stem}.json"
        save_json(save_path, record)
        print(f"Saved {save_path}")

    print(f"\nFinished. JSON output saved to {out_dir}")


if __name__ == "__main__":
    main()
