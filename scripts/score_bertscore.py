import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")

from bert_score import score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--lang", default="ru")
    args = ap.parse_args()

    data = json.load(open(args.inp))
    answers = data["answers"]
    refs = [a["reference"] for a in answers]
    hyps = [a["answer"] for a in answers]

    P, R, F1 = score(hyps, refs, lang=args.lang, verbose=False)
    bs_scores = F1.tolist()
    for a, bs in zip(answers, bs_scores):
        a["bert_score"] = bs

    avg = sum(bs_scores) / len(bs_scores)
    data["summary"]["avg_bert_score"] = avg

    with open(args.inp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"avg BERTScore F1: {avg:.4f}")

if __name__ == "__main__":
    main()
