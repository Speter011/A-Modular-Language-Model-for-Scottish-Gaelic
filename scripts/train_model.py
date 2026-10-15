from spacy.cli.train import train
from pathlib import Path

# Paths

#config_path = Path("../config.cfg")
config_path = Path("../config_coarse.cfg")
#output_dir = Path("../training/output")
output_dir = Path("../training/output_coarse")
#train_file = Path("../data/processed/train.spacy")
#dev_file = Path("../data/processed/dev.spacy")
#test_file = Path("../data/processed/test.spacy")


## Coarse files:
#train_file = Path("../data/processed_small_tagset/train.spacy")

# train different test sizes:
#train_file = Path("../data/processed_small_tagset/train_10.spacy")
#train_file = Path("../data/processed_small_tagset/train_20.spacy")
#train_file = Path("../data/processed_small_tagset/train_50.spacy")
#train_file = Path("../data/processed_small_tagset/train_75.spacy")
# Paths
data_dir = Path("../data/processed_small_tagset")
output_base = Path("../training/output_coarse")

dev_file = data_dir / "dev.spacy"
test_file = data_dir / "test.spacy"

# Train models using different training set sizes
train_sizes = [10, 20, 50, 75]

for size in train_sizes:
    print(f"\n{'='*40}")
    print(f"Training model with {size}% of training data")
    print(f"{'='*40}")

    # Training data for this wexperiment
    train_file = data_dir / f"train_{size}.spacy"

    # Separate output directory for each model
    output_dir = output_base / f"model-best_{size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train(
        config_path=config_path,
        output_path=output_dir,
        overrides={
            "paths.train": str(train_file),
            "paths.dev": str(dev_file),
            "training.max_epochs": 20
        }
    )

    print(f"Training finished. Model saved in {output_dir / 'model-best'}")

    # Load and evaluate model on the same test set
    import spacy
    from spacy.tokens import DocBin
    nlp = spacy.load(output_dir / "model-best")

    docbin = DocBin().from_disk(test_file)
    docs = list(docbin.get_docs(nlp.vocab))

    print(f"Loaded {len(docs)} test docs")

    for doc in docs[:3]:
        print([(t.text, t.tag_) for t in doc])



# dev_file = Path("../data/processed_small_tagset/dev.spacy")
# test_file = Path("../data/processed_small_tagset/test.spacy")

# output_dir.mkdir(exist_ok = True)


# # Training
# train(
#     config_path = config_path,
#     output_path = output_dir,
#     overrides={
#         "paths.train":str(train_file),
#         "paths.dev": str(dev_file),
#         "training.max_epochs": 20
#     }
# )

# print(f"Training finihsed. Model saved in {output_dir / 'model-best'}")    


# # load and test model
# import spacy
# from spacy.tokens import DocBin

# nlp = spacy.load(output_dir / "model-best")

# docbin = DocBin().from_disk(test_file)
# docs = list(docbin.get_docs(nlp.vocab))

# print(f"loaded {len(docs)} test docs")
# for doc in docs[:3]:
#     print([(t.text, t.tag_) for t in doc])