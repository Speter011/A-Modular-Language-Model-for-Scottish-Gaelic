import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from pprint import pprint
from spacy.scorer import Scorer
from collections import Counter


# load the model to evaluate
nlp_fine = spacy.load("../training/output/model-best")
nlp_coarse = spacy.load("../training/output_coarse/model-best")

# load test set
doc_bin = DocBin().from_disk("../data/processed_small_tagset/test.spacy")
#doc_bin = DocBin().from_disk("../data/processed_small_tagset/test.spacy")
gold_docs = list(doc_bin.get_docs(nlp_coarse.vocab))

examples = []

for gold_doc in gold_docs:
    pred_doc = nlp_coarse(gold_doc.text)
    examples.append(Example(pred_doc, gold_doc))

scores = Scorer().score(examples)

print(scores["tag_acc"])

pred_doc = nlp_coarse(gold_doc.text)
example = Example(pred_doc, gold_doc)

for gold_token, pred_token in zip(example.reference, example.predicted):
    print(gold_token.text, gold_token.tag_, pred_token.text, pred_token.tag_)
#classification not working!
#print(classification_report(y_true, y_pred, zero_division=0))

#print(len(y_true))
#print(len(y_pred))
#print(y_true[:20])
#print(y_pred[:20])

# weighted or macro f1, recall and precision
#p, r, f1,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
#print(p,r,f1)

# tags that are most common and were tagged most often
#print("Top true tags:", Counter(y_true).most_common())
#print("Top predicted tags:", Counter(y_pred).most_common())

#pprint(scores)