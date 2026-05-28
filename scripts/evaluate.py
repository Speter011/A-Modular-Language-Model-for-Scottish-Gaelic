import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training.example import Example
import json
from pathlib import Path

# Paths
#model_path = Path("../training/output_coarse/model-best")
model_path = Path("../training/output_coarse/model-best")
#test_file = Path("../data/processed/test.spacy")
test_file = Path("../data/processed_small_tagset/test.spacy")
#output_json = Path("../results/evaluation_metrics.json")
output_json = Path("../results/coarse_evaluation_metrics.json")

# load model
nlp = spacy.load(model_path)

docbin = DocBin().from_disk(test_file)
gold_docs = list(docbin.get_docs(nlp.vocab))
print(f"Loaded {len(gold_docs)} test docs")


# Evaluate
examples = []
for gold_doc in gold_docs:
    pred_doc = nlp(gold_doc.text)
    example = Example(pred_doc, gold_doc)
    examples.append(example)

    
scorer = Scorer()
metrics = scorer.score(examples)


print("\n--- Evaluation Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v}")

# Save output
if output_json:
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_json}")
