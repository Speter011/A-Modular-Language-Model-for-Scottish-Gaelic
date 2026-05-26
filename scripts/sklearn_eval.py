import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


# load the model to evaluate
nlp = spacy.load("./training/output/model-best")

# load test set
doc_bin = DocBin().from_disk("./data/processed/dev.spacy")
docs = list(doc_bin.get_docs(nlp.vocab))

y_true = []
y_pred = []


for doc in docs:
    pred_doc = nlp(doc.text)

    for gold_token, pred_token in zip(doc, pred_doc):
        y_true.append(gold_token.pos_)
        y_pred.append(pred_token.pos_)

print(classification_report(y_true, y_pred, zero_division=0))

#print(len(y_true))
#print(len(y_pred))
#print(y_true[:20])
#print(y_pred[:20])

# weighted or macro f1, recall and precision
p, r, f1,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

print(p,r,f1)

# tags that are most common and were tagged most often
#print("Top 20 true tags:", Counter(y_true).most_common(20))
#print("Top 20 predicted tags:", Counter(y_pred).most_common(20))
