"""Generate synthetic framed questions to approximate the missing 1,267 VQA-RAD
training pairs described in Lau et al. (2018).

Framed questions are standardized template versions of the original questions.
They preserve the same image and the same answer exactly.

Rules (per vqa_rad_dataset_debug.md):
- Only use train examples (phrase_type = freeform or para)
- Preserve image and answer exactly
- Mark new rows as phrase_type = synthetic_framed
- Do not touch the test set

Usage:
    cd biomedclip-repro/src
    python generate_framed_questions.py \
        --input  ../data/vqa_rad/VQA_RAD\ Dataset\ Public.json \
        --output ../data/vqa_rad/VQA_RAD_augmented.json
"""

import argparse
import json
import re


# ---------------------------------------------------------------------------
# Rule-based framer
# ---------------------------------------------------------------------------

def _strip(q: str) -> str:
    q = q.strip()
    if q and q[-1] in ".?!":
        q = q[:-1].strip()
    return q


def frame_question(question: str, answer: str, answer_type: str) -> str | None:
    """
    Rewrite question into a standardized template form.
    Returns the framed question string, or None if no safe rule applies.

    Design principle: each rule is conservative. When in doubt, return None
    so the example is skipped rather than corrupting the answer.
    """
    q = question.strip()
    ql = q.lower()
    ans_l = str(answer).lower().strip()

    # ------------------------------------------------------------------
    # YES / NO  (closed-ended)
    # ------------------------------------------------------------------
    # Starts with auxiliary verbs → "Is there evidence of X in this image?"
    yn_match = re.match(
        r'^(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\b',
        ql
    )
    # Also catch fragments without auxiliary: "any X?" / "any X present?"
    any_match = re.match(r'^any\b', ql)

    if yn_match or any_match:
        core = _strip(q)
        # Already a clean yes/no template — append "in this image?" if missing
        if not re.search(r'\b(image|scan|radiograph|film|study|picture|view)\b', ql, re.I):
            return core + " in this image?"
        else:
            return core + "?"

    # ------------------------------------------------------------------
    # WHAT
    # ------------------------------------------------------------------
    what_m = re.match(r'^what\b', ql)
    if what_m:
        core = _strip(q)
        # "What is X?" → "What is X in this image?"
        if not re.search(r'\b(image|scan|radiograph|film|study|picture|view|shown|pictured|present)\b', ql, re.I):
            return core + " in this image?"
        return core + "?"

    # ------------------------------------------------------------------
    # WHERE
    # ------------------------------------------------------------------
    if ql.startswith("where"):
        core = _strip(q)
        if not re.search(r'\b(image|scan|located|seen|visualized)\b', ql, re.I):
            return core + " located in this image?"
        return core + "?"

    # ------------------------------------------------------------------
    # HOW MANY / HOW MUCH
    # ------------------------------------------------------------------
    if re.match(r'^how (many|much)\b', ql):
        core = _strip(q)
        if not re.search(r'\b(image|scan|present|shown|visible|identified)\b', ql, re.I):
            return core + " are present in this image?"
        return core + "?"

    # ------------------------------------------------------------------
    # WHICH
    # ------------------------------------------------------------------
    if ql.startswith("which"):
        core = _strip(q)
        if not re.search(r'\b(image|scan|shown|imaged|present)\b', ql, re.I):
            return core + " in this image?"
        return core + "?"

    # ------------------------------------------------------------------
    # HOW  (non-"how many")
    # ------------------------------------------------------------------
    if ql.startswith("how"):
        core = _strip(q)
        return core + "?"

    # ------------------------------------------------------------------
    # "In what X" / "In which X"  (prepositional opening)
    # ------------------------------------------------------------------
    if re.match(r'^(in|on|at|to|from)\b', ql):
        core = _strip(q)
        return core + "?"

    # ------------------------------------------------------------------
    # "Describe X"
    # ------------------------------------------------------------------
    if re.match(r'^describe\b', ql):
        # "Describe X" → "What is the appearance of X in this image?"
        core = re.sub(r'^describe\s+', '', q, flags=re.I).strip().rstrip('?.')
        return f"What is the appearance of {core} in this image?"

    # ------------------------------------------------------------------
    # Bare noun / fragment that implies yes/no
    # e.g. "Pleural effusion present?"
    # ------------------------------------------------------------------
    if answer_type.upper() == "CLOSED":
        core = _strip(q)
        return f"Is there {core.lower()} in this image?"

    # No safe rule found
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="../data/vqa_rad/VQA_RAD Dataset Public.json")
    parser.add_argument("--output", default="../data/vqa_rad/VQA_RAD_augmented.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    train = [d for d in data if not d["phrase_type"].startswith("test_")]
    test  = [d for d in data if d["phrase_type"].startswith("test_")]

    generated, skipped = [], 0
    for d in train:
        framed = frame_question(d["question"], d["answer"], d.get("answer_type", "OPEN"))
        if framed is None:
            skipped += 1
            continue
        # Skip if framed == original (no actual change)
        if framed.strip().lower() == d["question"].strip().lower():
            skipped += 1
            continue
        row = dict(d)
        row["question"] = framed
        row["phrase_type"] = "synthetic_framed"
        row["original_question"] = d["question"]
        generated.append(row)

    augmented = data + generated
    with open(args.output, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"Original train:    {len(train)}")
    print(f"Generated framed:  {len(generated)}  (skipped {skipped})")
    print(f"Augmented train:   {len(train) + len(generated)}")
    print(f"Test (unchanged):  {len(test)}")
    print(f"Saved to {args.output}")

    # Spot-check (generated rows carry original_question for alignment)
    print("\nSample rewrites:")
    for g in generated[:8]:
        print(f"  [{g['answer_type']}] {g['original_question']}")
        print(f"       -> {g['question']}")
        print(f"       answer: {g['answer']}")


if __name__ == "__main__":
    main()
