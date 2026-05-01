import argparse
import glob
import importlib.util
import os
import sys
from pathlib import Path


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    module_dir = repo_root / "model"
    florence_path = module_dir / "florenceAutolabeling.py"
    qwen_path = module_dir / "Qwen3.5labeling.py"

    florence = load_module_from_path("florenceAutolabeling", florence_path)
    qwen = load_module_from_path("qwen3_5_labeling", qwen_path)

    parser = argparse.ArgumentParser(
        description="Run Florence autolabeling first, then Qwen3.5 JSON risk label updates."
    )
    parser.add_argument(
        "--image_dir", type=str, default="./dataset/test/images",
        help="Directory containing test images for Florence labeling.",
    )
    parser.add_argument(
        "--label_dir", type=str, default="./dataset/test/labels",
        help="Directory for JSON labels that Florence writes and Qwen updates.",
    )
    parser.add_argument(
        "--model_path", type=str,
        default=None,
        help=(
            "Florence-2 model path or HuggingFace repo slug. "
            "If omitted, uses the default from florenceAutolabeling.py."
        ),
    )
    parser.add_argument(
        "--max_tokens", type=int, default=256,
        help="Max tokens for Florence-2 inference.",
    )
    parser.add_argument(
        "--omlx_endpoint", type=str,
        default=os.environ.get("OMLX_ENDPOINT", "http://127.0.0.1:8000"),
        help="oMLX server endpoint for Qwen3.5.",
    )
    parser.add_argument(
        "--qwen_model_id", type=str,
        default=os.environ.get(
            "QWEN_MODEL_ID",
            "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        ),
        help="Model ID to use for Qwen3.5 classification.",
    )
    parser.add_argument(
        "--qwen_max_tokens", type=int, default=10,
        help="Max tokens for Qwen3.5 classification responses.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Run the Qwen update pass without writing JSON files.",
    )
    parser.add_argument(
        "--skip_florence", action="store_true",
        help="Skip Florence autolabeling and only run Qwen3.5 updates.",
    )
    parser.add_argument(
        "--skip_qwen", action="store_true",
        help="Run only Florence autolabeling and skip Qwen3.5 updates.",
    )
    parser.add_argument(
        "--test_omlx", action="store_true",
        help="Test the oMLX server connection before updating labels.",
    )
    args = parser.parse_args()

    args.label_dir = os.path.abspath(args.label_dir)
    args.image_dir = os.path.abspath(args.image_dir)

    os.makedirs(args.label_dir, exist_ok=True)

    if not args.skip_florence:
        model_path = args.model_path
        if model_path is None:
            model_path = getattr(
                florence,
                "DEFAULT_FLORENCE2_LOCAL_PATH",
                None,
            ) or getattr(
                florence,
                "DEFAULT_FLORENCE2_MODEL_ID",
                None,
            )

        print("\n=== Florence Autolabeling Pass ===")
        print(f"Image dir : {args.image_dir}")
        print(f"Label dir : {args.label_dir}")
        print(f"Model path: {model_path}")
        florence_backend = florence.load_florence2(model_path)
        print(f"Florence backend: {florence_backend}\n")
        florence.label_pet_risk_directory(
            args.image_dir,
            args.label_dir,
            max_tokens=args.max_tokens,
        )

    if args.skip_qwen:
        print("\nSkipping Qwen3.5 update pass.")
        return

    if args.test_omlx:
        ok = qwen.test_connection(args.omlx_endpoint, args.qwen_model_id)
        if not ok:
            raise SystemExit(1)

    print("\n=== Qwen3.5 Risk Label Update Pass ===")
    print(f"Label dir      : {args.label_dir}")
    print(f"oMLX endpoint  : {args.omlx_endpoint}")
    print(f"Qwen model ID  : {args.qwen_model_id}")
    print(f"Dry run        : {args.dry_run}\n")

    json_paths = sorted(glob.glob(os.path.join(args.label_dir, "*.json")))
    if not json_paths:
        raise FileNotFoundError(f"No JSON label files found in {args.label_dir}")

    for path in json_paths:
        qwen.process_label_file(
            path,
            args.omlx_endpoint,
            args.qwen_model_id,
            args.qwen_max_tokens,
            args.dry_run,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
