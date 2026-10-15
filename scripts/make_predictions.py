
from pathlib import Path
import csv
import json
from collections import Counter

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer

# Paths

model_path = Path("../training/output_coarse/model-best")
test_file = Path("../data/processed_small_tagset/test.spacy")

results_dir = Path("../results")
results_dir.mkdir(parents=True, exist_ok=True)

predictions_file = results_dir / "coarse_full_predictions.csv"
metrics_file = results_dir / "coarse_full_evaluation_metrics.json"


# Load model and gold-standard test documents
nlp = spacy.load(model_path)

doc_bin = DocBin().from_disk(test_file)
gold_docs = list(doc_bin.get_docs(nlp.vocab))

rows = []
examples = []

correct = 0
total = 0
gold_counts = Counter()
pred_counts = Counter()
pair_counts = Counter()


# Evaluate each document
for doc_id, gold_doc in enumerate(gold_docs):

    # Predict on the original text, without copying gold tags.
    pred_doc = nlp(gold_doc.text)

    # Build an Example for spaCy's scorer.
    example = Example(pred_doc, gold_doc)
    examples.append(example)

    # Align tokens by their character offsets.
    # This allows us to identify tokenization differences.
    alignment = example.alignment

    for pred_token in pred_doc:
        aligned_gold_indices = alignment.x2y[pred_token.i]

        # Only compare tokens with exactly one matching
        # gold token. Record alignment problems separately.
        if len(aligned_gold_indices) != 1:
            rows.append({
                "doc_id": doc_id,
                "token_index": pred_token.i,
                "token": pred_token.text,
                "gold_tag": "",
                "predicted_tag": pred_token.tag_,
                "correct": "",
                "alignment_status": "not_one_to_one",
                "sentence": gold_doc.text,
            })
            continue

        gold_token = gold_doc[aligned_gold_indices[0]]

        gold_tag = gold_token.tag_
        predicted_tag = pred_token.tag_

        is_correct = gold_tag == predicted_tag

        total += 1
        correct += int(is_correct)

        gold_counts[gold_tag] += 1
        pred_counts[predicted_tag] += 1
        pair_counts[(gold_tag, predicted_tag)] += 1

        rows.append({
            "doc_id": doc_id,
            "token_index": pred_token.i,
            "token": pred_token.text,
            "gold_tag": gold_tag,
            "predicted_tag": predicted_tag,
            "correct": is_correct,
            "alignment_status": "aligned",
            "sentence": gold_doc.text,
        })



# Save token-level predictions
fieldnames = [
    "doc_id",
    "token_index",
    "token",
    "gold_tag",
    "predicted_tag",
    "correct",
    "alignment_status",
    "sentence",
]

with predictions_file.open("w", encoding="utf-8-sig",
                           newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)



# Independent metrics
accuracy = correct / total if total else 0.0

labels = sorted(set(gold_counts) | set(pred_counts))
per_tag = {}

for tag in labels:
    # A true positive is a correct prediction of this tag.
    tp = pair_counts[(tag, tag)]

    # Predicted this tag, but the gold tag was different.
    fp = sum(
        count for (gold, pred), count in pair_counts.items()
        if pred == tag and gold != tag
    )

    # Gold tag was this tag, but another tag was predicted.
    fn = sum(
        count for (gold, pred), count in pair_counts.items()
        if gold == tag and pred != tag
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    per_tag[tag] = {
        "support": gold_counts[tag],
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# spaCy's scores, using the same predicted/reference Examples.
spacy_scores = Scorer(nlp).score(examples)

metrics = {
    "model": str(model_path),
    "test_file": str(test_file),
    "n_documents": len(gold_docs),
    "n_aligned_tokens": total,
    "n_correct_tags": correct,
    "accuracy": accuracy,
    "per_tag": per_tag,
    "confusion_pairs": {
        f"{gold} -> {pred}": count
        for (gold, pred), count in sorted(pair_counts.items())
    },
    "spacy_scores": spacy_scores,
}

with metrics_file.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"Predictions saved to: {predictions_file}")
print(f"Metrics saved to: {metrics_file}")
print(f"Independent aligned-token accuracy: {accuracy:.4%}")
print(f"spaCy tag accuracy: {spacy_scores.get('tag_acc')}")
print(f"spaCy token accuracy: {spacy_scores.get('token_acc')}")
print(f"Aligned tokens compared: {total}")
print(f"Documents evaluated: {len(gold_docs)}")
