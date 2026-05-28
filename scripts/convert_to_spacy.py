import os
import re
import spacy
from spacy.tokens import Doc, DocBin
from spacy.vocab import Vocab
from pathlib import Path
import random


# Paths
raw_dir = Path("../data/raw_small_tagset") #changed from "../data/raw"
map_file = raw_dir / "gd-parole.map" 
print(map_file)
out_dir = Path("../data/processed_small_tagset") #changed from "../data/processed"
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


# use a new blank model for Gaelic
nlp = spacy.blank("gd")
vocab = nlp.vocab


def parse_arcosg_file(path):
    """Parse the file given. 
    Convert it to doc(vocab, words, spaces, tags, pos, sent_starts) format.

    Args:
        path(str): File path to raw, but annotated data.

    Returns:
        Doc: doc formatted data to be used for training and testing in spaCy models.
    """
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    ### Fine grained PoS training ###
    # remove undescores, capitalize all pos tags
    #words = [re.sub("_", " ", w) for w in re.findall(r"([^ \n]+?)/", text)]
    #tags = [t.upper() for t in re.findall(r"/([^ \n]*)", text)]

    # if a tag is not in the tagset replace with the default "X" for standardization
    #pos = [pos_map.get(t, "X") for t in tags]

    ### Coarse PoS training ###
    pairs = re.findall(r"([^/\s]+)/([^/\s]+)", text)

    words = []
    fine_tags = []

    # 1.a create word and tag pairs and remove underscore
    for w, t in pairs:
        w = w.replace("_", " ")
        t = t.upper()

        # drop speaker tokens
        if t == "XSC":
            continue
        if t == 'X':
            continue
        if re.match(r"^\[\d+\]$", w):
            continue
        
        #add the cleaned pairs to the sets
        words.append(w)
        fine_tags.append(t)


    # add coarse pos tags as well from the map file.
    #tags = [pos_map.get(t, "X") for t in fine_tags] # this is broken when using the coarse set as most are replaced with just X
    tags = fine_tags


    # 1.b throw error if the number of words and tags don't match
    if len(words) != len(fine_tags):
        raise ValueError(f"Number of Words ({len(words)}) != number of tags ({len(fine_tags)}) in {path.name}")
    # throw errror for empty words and tags
    if any(t == '' for t in fine_tags):
        raise ValueError(f"Empty tag detected in {path.name}")
    if any(w == '' for w in words):
        raise ValueError(f"Empty word detected in {path.name}")

    
    # 2. spaces after token (Fq and Fz are opening and closing quotation marks)
    spaces = []
    for i, word in enumerate(words):
        space = True
        if word.endswith("-"):
            space = False
        elif i != len(words)-1 and words[i+1] in [".", ",", ":", ";", "?", "!", "-", "_", ")", "—"]:
            space = False            
        elif fine_tags[i] == "FQ":   #replaced from tags[i]
            space = False
        elif i != len(words)-1 and tags[i+1] == "Fz":
            space = False
        elif word == "(":
            space = False
        spaces.append(space)


    # 3. sentence starts
    sent_starts =[]
    for i in range(len(words)):
        if i == 0:
            sent_starts.append(True)
        elif fine_tags[i - 1] == "XSC":  # just to make sure
            sent_starts.append(True)
        elif words[i - 1] in [".", "?", "!"]:
            sent_starts.append(True)
        else:
            sent_starts.append(False)
    
    # sent_starts = [True]
    # for i in range(1, len(words)):
    #     if fine_tags[i-1] == "XSC" or words[i-1] in [".", "?", "!"]: #replaced from tags[i-1]
    #         sent_starts.append(True)
    #     else:
    #         sent_starts.append(False)

    # Remove speaker designations (Xsc tag refers to a speaker in transcripts such as [1] indicating the first speaker)
    # for i in range(len(words)-1, -1, -1):
    #     if fine_tags[i] == "XSC": #replaced from tags[i]
    #         words.pop(i)
    #         fine_tags.pop(i) #added for coarse tag experiment
    #         tags.pop(i)
    #         #pos.pop(i)
    #         spaces.pop(i)
    #         sent_starts.pop(i)

    # 4. Create Doc
    doc = Doc(vocab = vocab,
              words = words, 
              spaces = spaces,
              tags = tags,
              #pos = tags,
              sent_starts = sent_starts)
    
    ## DeBugging:
    from collections import Counter
    print(Counter(tags).most_common(20))
    print("Num unique tags:", len(set(tags)))

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

# Split sets
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