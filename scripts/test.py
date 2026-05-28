# load and test model
import spacy
from spacy.tokens import DocBin
from pathlib import Path
test_file = Path("../data/processed_small_tagset/test.spacy")

nlp = spacy.load("../training/output_coarse/model-best")

docbin = DocBin().from_disk(test_file)
docs = list(docbin.get_docs(nlp.vocab))

print(f"loaded {len(docs)} test docs")
for doc in docs[:3]:
    print([(t.text, t.tag_) for t in doc])