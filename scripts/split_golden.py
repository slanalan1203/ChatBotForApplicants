import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
DEV_FRACTION = 0.7
GOLDEN = Path("data/golden_set.jsonl")
DEV_OUT = Path("data/golden_dev.jsonl")
TEST_OUT = Path("data/golden_test.jsonl")

def main():
    rng = random.Random(SEED)
    data = [json.loads(l) for l in GOLDEN.open()]
    buckets = defaultdict(list)
    for d in data:
        key = (d["category"], d["difficulty"])
        buckets[key].append(d)

    dev, test = [], []
    for key, items in buckets.items():
        rng.shuffle(items)
        n_dev = round(len(items) * DEV_FRACTION)
        dev.extend(items[:n_dev])
        test.extend(items[n_dev:])

    rng.shuffle(dev)
    rng.shuffle(test)

    with DEV_OUT.open("w") as f:
        for d in dev:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with TEST_OUT.open("w") as f:
        for d in test:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"dev: {len(dev)}, test: {len(test)}")
    cats = sorted(set(d["category"] for d in data))
    print(f"{'category':<15} {'dev':>5} {'test':>5}")
    for c in cats:
        nd = sum(1 for d in dev if d["category"] == c)
        nt = sum(1 for d in test if d["category"] == c)
        print(f"{c:<15} {nd:>5} {nt:>5}")

if __name__ == "__main__":
    main()
