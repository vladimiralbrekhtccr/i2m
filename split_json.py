import json
import random

INPUT_JSON = "large_files/ILSVRC/meta_files/pairs.json"
TRAIN_OUT = "train.json"
EVAL_OUT = "eval.json"
TRAIN_RATIO = 0.9
SEED = 42

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Expected JSON to be a list of {image, audio} objects")

    for i, it in enumerate(items[:5]):
        if "image" not in it or "audio" not in it:
            raise ValueError(f"Bad format at index {i}: {it}")

    random.seed(SEED)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * TRAIN_RATIO)

    train_items = items[:n_train]
    eval_items = items[n_train:]

    print(f"Total: {n}, Train: {len(train_items)}, Eval: {len(eval_items)}")

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train_items, f, indent=2)

    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(eval_items, f, indent=2)

if __name__ == "__main__":
    main()
