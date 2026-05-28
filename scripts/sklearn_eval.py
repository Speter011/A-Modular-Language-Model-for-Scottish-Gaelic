import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report 
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter


# load the model to evaluate
nlp_fine = spacy.load("../training/output/model-best")
nlp_coarse = spacy.load("../training/output_coarse/model-best")

# load test set
#doc_bin = DocBin().from_disk("./data/processed/dev.spacy")
doc_bin = DocBin().from_disk("../data/processed_small_tagset/test.spacy")
gold_docs = list(doc_bin.get_docs(nlp_coarse.vocab))

y_true = []
y_pred = []

for gold_doc in gold_docs:
    pred_doc = nlp_coarse(gold_doc)

    for gold_token, pred_token in zip(gold_doc, pred_doc):
        y_true.append(gold_token.tag_)
        y_pred.append(pred_token.tag_)

    # token alignment check (important!)
    if len(pred_doc) != len(gold_doc):
        print("There is a problem!!!!")
        continue  # or handle properly if this happens often
#labels = sorted(list(set(y_true)))

print("Total tokens evaluated:", len(y_true))
print("Unique POS tags:",len(set(y_true)))
print(classification_report(y_true, y_pred, zero_division=0))

#print(len(y_true))
#print(len(y_pred))
#print(y_true[:20])
#print(y_pred[:20])

# weighted or macro f1, recall and precision
#p, r, f1,_ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
#print(p,r,f1)

# tags that are most common and were tagged most often
#print("Top 20 true tags:", Counter(y_true).most_common(20))
#print("Top 20 predicted tags:", Counter(y_pred).most_common(20))