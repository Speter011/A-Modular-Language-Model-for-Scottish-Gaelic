from spacy.cli.train import train
from pathlib import Path

# Paths

config_path = Path("../config.cfg")
output_dir = Path("../training/output")
train_file = Path("../data/processed/train.spacy")
dev_file = Path("../data/processed/dev.spacy")
test_file = Path("../data/processed/test.spacy")

output_dir.mkdir(exist_ok = True)


# Training
train(
    config_path = config_path,
    output_path = output_dir,
    overrides={
        "paths.train":str(train_file),
        "paths.dev": str(dev_file),
        "training.max_epochs": 20
    }
)

print(f"Training finihsed. Model saved in {output_dir / 'model-best'}")


# load and test model
import spacy
from spacy.tokens import DocBin

nlp = spacy.load(output_dir / "model-best")

docbin = DocBin().from_disk(test_file)
docs = list(docbin.get_docs(nlp.vocab))

print(f"loaded {len(docs)} test docs")
for doc in docs[:3]:
    print([(t.text, t.tag_) for t in doc])