import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.runner import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden_dev.jsonl")
    ap.add_argument("--out", default="data/eval/last_run.json")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    summary, per_q = evaluate(Path(args.golden))
    if args.label:
        summary["label"] = args.label
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"summary": summary, "per_q": per_q}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
