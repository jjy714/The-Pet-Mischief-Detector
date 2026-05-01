import os
import glob
import json
import argparse
import re
import urllib.request
import urllib.error

DEFAULT_OMLX_ENDPOINT = os.environ.get("OMLX_ENDPOINT", "http://127.0.0.1:8000")
DEFAULT_MODEL_ID = os.environ.get(
    "QWEN_MODEL_ID",
    "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
)

LABEL_KEY = "risk_level"
REASON_KEY = "reason"
VALID_LEVELS = {"HIGH", "MEDIUM", "LOW"}

# Made the system prompt much more robotic and strict
SYSTEM_PROMPT = (
    "You are a strict, automated risk classification system. You have no personality. "
    "You must output EXACTLY two lines. "
    "Line 1: ONLY the word HIGH, MEDIUM, or LOW. "
    "Line 2: A short 3-8 word reason. "
    "Never use conversational filler, never say 'Let me analyze', and never list scene elements."
)

SINGLE_LABEL_SYSTEM = (
    "You are a strict risk classification assistant. "
    "Read the text below and output only one word: HIGH, MEDIUM, or LOW. "
    "Do not output any explanation, analysis, or extra words."
)

SINGLE_REASON_SYSTEM = (
    "You are a strict risk summary assistant. "
    "Read the text below and output a single short reason phrase for the risk level. "
    "Do not output the label, explanation, or any extra text."
)


def call_omlx(endpoint: str, model_id: str, messages: list[dict], max_tokens: int = 40, stop: list[str] | None = None) -> str:
    """Send a chat completion request to the local oMLX server."""
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    if stop is not None:
        payload["stop"] = stop

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


def test_connection(endpoint: str, model_id: str) -> bool:
    """Ping the oMLX server and run a minimal smoke-test generation."""
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

    print(f"\nSending smoke-test prompt to model '{model_id}' …")
    try:
        result = call_omlx(
            endpoint,
            model_id,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Reply with exactly one word: HIGH. Do not explain."},
            ],
            max_tokens=5,
        )
        print(f"✅  Model responded: '{result}'")
    except urllib.error.URLError as e:
        print(f"❌  Request failed: {e.reason}")
        return False
    except Exception as e:
        print(f"❌  Unexpected error: {e}")
        return False

    print("\n--- Test complete ---\n")
    return True


def build_prompt(raw_text: str) -> str:
    # Restructured to put the rules first, then the description, and ending with a rigid format template
    return (
        "Evaluate the following scene for pet-object interaction risk.\n\n"
        "RULES:\n"
        "- LOW: pet far from object or clearly inactive.\n"
        "- MEDIUM: pet near object, possible interaction but no clear action.\n"
        "- HIGH: clear imminent interaction (approaching fast, touching, manipulating).\n\n"
        "SCENE DESCRIPTION:\n"
        f"{raw_text.strip()}\n\n"
        "OUTPUT FORMAT (Exactly two lines):\n"
        "Line 1: [HIGH/MEDIUM/LOW]\n"
        "Line 2: [Short reason]\n\n"
        "YOUR PREDICTION:"
    )


def _extract_plain_text(raw_text: str) -> str | None:
    if "GenerationResult(" in raw_text and "text=" in raw_text:
        match = re.search(r"GenerationResult\(text=('|\")(.*?)(?:\1)(?:,|\))", raw_text, re.DOTALL)
        if match:
            cleaned = match.group(2).strip()
            cleaned = re.sub(r'^(?:<s>)+', '', cleaned)
            cleaned = re.sub(r'(?:<s>)+$', '', cleaned)
            cleaned = cleaned.strip()
            if cleaned and re.search(r'[A-Za-z0-9]', cleaned):
                return cleaned
            return None
    cleaned = raw_text.strip()
    if cleaned and re.search(r'[A-Za-z0-9]', cleaned):
        return cleaned
    return None


def _parse_label_and_reason(prediction: str) -> tuple[str | None, str | None]:
    text = prediction.strip()
    if not text:
        return None, None

    match = re.search(r'\b(HIGH|MEDIUM|LOW)\b', text, re.IGNORECASE)
    if match:
        label = match.group(1).upper()
        remainder = text[match.end():].strip(" \n:.-")
        reason = None
        if remainder:
            lines = [line.strip() for line in remainder.split('\n') if line.strip()]
            if lines:
                reason = lines[0]
        return label, reason

    return None, None


HIGH_PATTERNS = [
    r'\b(touch|touching|touches|manipulat|grab|grabbing|chew|chewing|bite|jump|jumping|climb|climbing|push|pushing|drag|dragging|press|pressing)\b',
    r'\b(on|onto|in|onto|onto)\s+(keyboard|laptop|computer|monitor|desk|table|counter|bed|sofa|couch|chair)\b',
    r'\b(sitting|lying|resting)\s+on\s+(keyboard|laptop|computer|monitor|desk|table|bed|sofa|couch|chair)\b',
    r'\b(may|might|could)\s+touch\b',
    r'\b(about\s+to|imminent|soon|approaching|reaching)\b',
]
MEDIUM_PATTERNS = [
    r'\b(near|next to|beside|close to|nearby|by|adjacent to|alongside)\b',
    r'\b(possible interaction|possible to interact|may interact|might interact|could interact)\b',
    r'\b(near|next to|beside)\s+(object|plant|box|bag|bottle|pot|fence|keyboard|laptop|computer|monitor|desk|table|bed|sofa|couch|chair)\b',
]
LOW_PATTERNS = [
    r'\b(sleep|sleeping|lying|resting|watching|looking up|looked up|looking at)\b',
    r'\b(far from|away from|distant from)\b',
]


def rule_based_label(text: str) -> str:
    normalized = text.lower()
    for pattern in HIGH_PATTERNS:
        if re.search(pattern, normalized):
            return "HIGH"
    for pattern in MEDIUM_PATTERNS:
        if re.search(pattern, normalized):
            return "MEDIUM"
    return "LOW"


def extract_reason_from_analysis(prediction: str) -> str | None:
    text = prediction.lower()
    if re.search(r'not interacting|no interaction|not interacting with', text):
        return "pet not interacting with object"
    if re.search(r'near.*(keyboard|laptop|computer|desk|table|bed|sofa|couch|chair|plant|box|bag|bottle|pot|fence)', text):
        return "pet near object, possible interaction"
    if re.search(r'(may|might|could) touch|about to touch|approaching|reaching|touching|manipulating|grabbing|chewing|pushing|pressing', text):
        return "clear imminent interaction"
    if re.search(r'sleeping|lying|resting|watching|looking up|looking at', text):
        return "pet far from object or clearly inactive"
    return None


def rule_based_reason(text: str) -> str:
    label = rule_based_label(text)
    if label == "HIGH":
        return "clear imminent interaction"
    if label == "MEDIUM":
        return "pet near object, possible interaction"
    return "pet far from object or clearly inactive"


def _classify_from_analysis(endpoint: str, model_id: str, analysis: str, max_tokens: int = 10) -> str | None:
    messages = [
        {"role": "system", "content": SINGLE_LABEL_SYSTEM},
        {"role": "user", "content": f"Analysis:\n{analysis}\n\nLabel:"},
    ]
    try:
        result = call_omlx(endpoint, model_id, messages, max_tokens=max_tokens)
    except Exception:
        return None
    label, _ = _parse_label_and_reason(result)
    return label


def _summarize_reason(endpoint: str, model_id: str, analysis: str, max_tokens: int = 20) -> str | None:
    messages = [
        {"role": "system", "content": SINGLE_REASON_SYSTEM},
        {"role": "user", "content": f"Analysis:\n{analysis}\n\nReason:"},
    ]
    try:
        result = call_omlx(endpoint, model_id, messages, max_tokens=max_tokens)
    except Exception:
        return None
    return result.strip()


def classify_text(endpoint: str, model_id: str, raw_text: str, max_tokens: int = 40) -> tuple[str, str] | None:
    # Increased default max_tokens slightly to 40 to give it breathing room if it slips up
    cleaned_text = _extract_plain_text(raw_text)
    if not cleaned_text:
        print("  ⚠️  Raw description invalid or empty; skipping classification")
        return None
        
    prompt = build_prompt(cleaned_text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        prediction = call_omlx(endpoint, model_id, messages, max_tokens=max_tokens)
    except Exception as e:
        print(f"  ⚠️  API call failed: {e} — skipping update")
        return None

    label, reason = _parse_label_and_reason(prediction)

    if label is None:
        label = rule_based_label(cleaned_text)
    if reason is None:
        reason = extract_reason_from_analysis(prediction)
    if reason is None:
        reason = _summarize_reason(endpoint, model_id, prediction, max_tokens=20)
    if reason is None:
        reason = rule_based_reason(cleaned_text)
    return label, reason or ""

    # Strict retry if the first attempt completely failed to include a label
    strict_prompt = (
        prompt +
        "\n\nFATAL ERROR: You failed to provide a classification. "
        "Respond NOW with EXACTLY one word (HIGH, MEDIUM, or LOW) on the first line."
    )
    try:
        retry_prediction = call_omlx(
            endpoint,
            model_id,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": strict_prompt},
            ],
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"  ⚠️  Retry API call failed: {e} — skipping update")
        print(f"  ⚠️  Invalid classification output: '{prediction}' — skipping update")
        return None

    label, reason = _parse_label_and_reason(retry_prediction)
    if label is not None:
        if reason is None:
            reason = _summarize_reason(endpoint, model_id, retry_prediction, max_tokens=20)
        return label, reason or ""

    print(
        f"  ⚠️  Invalid classification output: '{prediction}' — retry invalid: '{retry_prediction}' — skipping update"
    )
    return None


def load_label_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_label_file(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_label_file(path: str, endpoint: str, model_id: str, max_tokens: int, dry_run: bool):
    data = load_label_file(path)
    raw_text = data.get("raw", "")
    if not raw_text:
        print(f"Skipping {os.path.basename(path)} — raw text is empty.")
        return

    result = classify_text(endpoint, model_id, raw_text, max_tokens)
    if result is None:
        print(f"Skipping {os.path.basename(path)} — classification result invalid.")
        return

    new_level, new_reason = result
    old_level = data.get(LABEL_KEY, "UNKNOWN")
    old_reason = data.get(REASON_KEY, "")
    if old_level == new_level and old_reason == new_reason:
        print(f"{os.path.basename(path)}: already {new_level}")
        return

    data[LABEL_KEY] = new_level
    data[REASON_KEY] = new_reason
    print(f"{os.path.basename(path)}: {old_level} -> {new_level}; reason: {new_reason}")
    if not dry_run:
        save_label_file(path, data)


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 JSON risk label updater via oMLX")
    parser.add_argument(
        "--label_dir", type=str, default="./dataset/test/labels",
        help="Directory containing JSON label files",
    )
    parser.add_argument(
        "--omlx_endpoint", type=str, default=DEFAULT_OMLX_ENDPOINT,
        help="oMLX server URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--model_id", type=str, default=DEFAULT_MODEL_ID,
        help="Model ID as shown in oMLX or Hugging Face",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=40,
        help="Max tokens for the classification response",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show updates without writing files",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Single JSON file to update instead of the whole directory",
    )
    parser.add_argument(
        "--test-connection", action="store_true",
        help="Ping the oMLX server and run a smoke-test, then exit",
    )
    args = parser.parse_args()

    if args.test_connection:
        ok = test_connection(args.omlx_endpoint, args.model_id)
        raise SystemExit(0 if ok else 1)

    if args.file:
        paths = [args.file]
    else:
        paths = sorted(glob.glob(os.path.join(args.label_dir, "*.json")))

    if not paths:
        raise FileNotFoundError(f"No JSON files found in {args.label_dir}")

    for path in paths:
        process_label_file(
            path,
            args.omlx_endpoint,
            args.model_id,
            args.max_new_tokens,
            args.dry_run,
        )


if __name__ == "__main__":
    main()