import os
import re
import spacy
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from pathlib import Path
import random


# Paths
raw_dir = Path("../data/raw")
map_file = raw_dir / "gd-parole.map"
out_dir = Path("../data/processed")
out_dir.mkdir(parents = True, exist_ok = True)


# Load PoS map (these tags are different from the UD formatting)
pos_map = {}
with open(map_file, "r", encoding="utf-8") as f:
    for line in f:
        key, value = line.strip().split()
        if value == ".":
            value = "PUNCT"
        elif value == "PRT":
            value = "PART"
        pos_map[key] = value


nlp = spacy.blank("gd")
vocab = nlp.vocab

def parse_arcosg_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    words = [re.sub("_", " ", w) for w in re.findall(r"([^ \n]+?)/", text)]
    tags = [t.upper() for t in re.findall(r"/([^ \n]*)", text)]
    pos = [pos_map.get(t, "X") for t in tags]

    if len(words) != len(tags):
        raise ValueError(f"Number of Words ({len(words)}) != number of tags ({len(tags)}) in {path.name}")
    
    # spaces after token
    spaces = []
    for i, word in enumerate(words):
        space = True
        if word.endswith("-"):
            space = False
        elif i != len(words)-1 and words[i+1] in [".", ",", ":", ";", "?", "!", "-", "_", ")", "—"]:
            space = False            
        elif tags[i] == "Fq":
            space = False
        elif i != len(words)-1 and tags[i+1] == "Fz":
            space = False
        elif word == "(":
            space = False
        spaces.append(space)

    # sentence starts
    sent_starts = [True]
    for i in range(1, len(words)):
        if tags[i-1] == "Xsc" or words[i-1] in [".", "?", "!"]:
            sent_starts.append(True)
        else:
            sent_starts.append(False)

    # Remove speaker designations
    for i in range(len(words)-1, -1, -1):
        if tags[i] == "Xsc":
            words.pop(i)
            tags.pop(i)
            pos.pop(i)
            spaces.pop(i)
            sent_starts.pop(i)

    # Create Doc
    doc = Doc(vocab = vocab,
              words = words, 
              spaces = spaces,
              tags = tags,
              pos = pos,
              sent_starts = sent_starts)
    
    return doc

# main processing
docs = []

for file in raw_dir.glob("*.txt"):
    print(f"Processing {file.name}...")
    doc = parse_arcosg_file(file)
    docs.append(doc)

print(f"Total documents: {len(docs)}")

# shuffle
random.seed(10)
random.shuffle(docs)

# Split
train = 0.8
dev = 0.1
test = 0.1

n = len(docs)
n_train = int(n * train)
n_dev = int(n * dev)

train_docs = docs[:n_train]
dev_docs = docs[n_train: n_train + n_dev]
test_docs = docs[n_train + n_dev:]

# Save DocBins
DocBin(docs=train_docs).to_disk(out_dir / "train.spacy")
DocBin(docs=dev_docs).to_disk(out_dir / "dev.spacy")
DocBin(docs=test_docs).to_disk(out_dir / "test.spacy")

print(f"Saved train/dev/test splits to {out_dir}")
print(f"Train split:{train:.0%}")
print(f"Dev split:{dev:.0%}")
print(f"Test split:{1 - train - dev:.0%}")