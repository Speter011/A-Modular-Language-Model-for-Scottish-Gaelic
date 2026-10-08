from pathlib import Path
import spacy
from spacy.tokens import DocBin

# data folders
project_root = Path(__file__).resolve().parent.parent
clean_folder = project_root / "data" / "clean"


# load in the model to test
#nlp = spacy.load("gd_core_arcosg_sm")
nlp = spacy.load("../training/output_coarse/model-best")
print(nlp.pipe_names)

#doc = nlp("Tha mi a' dol dhan sgoil an-diugh.") # I am going to school today

#for token in doc:
#    print(token.text, token.pos_, token.tag_)

# texts = [
# "Tha mi toilichte an-diugh.",
# "Chuala mi sgeulachd bho sheann duine anns a' bhaile.",
# "Bha an cù beag a' ruith tron phàirc.",
# "Thèid sinn dhan sgoil a-màireach.",
# "Carson a tha i cho brònach?",
# "Chunnaic Iain an càr dearg aig an taigh mhòr.",
# "Chan eil mi a' tuigsinn an leasan seo.",
# "Bha na daoine a' bruidhinn gu luath agus gu sunndach.",
# "Tha Alba na dùthaich bhrèagha.",
# "Am faca tu am film ùr a-raoir?",
# "Ghabh e biadh anns a' chidsin mhòr.",
# "Bha e glè fhuar ach bha a' ghrian a' deàrrsadh.",
# "Thàinig iad dhachaigh an dèidh na h-obrach.",
# "Tha mi a' smaoineachadh gu bheil e ceart.",
# "Càite an deach an leabhar agam?"
# ]

# grammatically difficult backtranslated and generated texts
# texts = [
#     "An-dè, bha mi ann an taigh mo sheanmhar-sa air iomall a bhaile, far an robh i fhèin agus a nighean-an-nighean a' dèanamh biadh airson na Nollaige."
#     "Bha an taigh beag, ach bha e làn de sheann leabhraichean, dealbhan-camara agus rudan nach robh mi air fhaicinn roimhe. "
#     "“Nach eil thu sgìth?” dh'fhaighnich i rium, nuair a thàinig mi a-steach. “Tha thu air a bhith ag obair fad an latha!” Thuirt mi rithe nach robh, agus gun robh mi airson cuideachadh."
#     "Thug i dhomh sgian agus thuirt i: “Gearr na glasraich seo—na fàg iad ro-mhòr.” Bha mi a' feuchainn ri èisteachd rithe, ach bha mo bhràthair-sa anns an t-seòmar eile agus e a' cluich ceòl ro-àrd. "
#     "An ceann greis, thàinig nàbaidh a-steach. “Ciamar a tha sibh?” thuirt e. Bha e air a bhith a' fuireach ann an Glaschu fad fichead bliadhna, ach bha e fhathast a' tighinn dhachaigh gach geamhradh. Thuirt e gun robh e airson seann taigh a theaghlaich a chàradh—taigh a bha air a bhith falamh fad iomadh bliadhna."
#     "Bha sinn uile a' bruidhinn mu dheidhinn, agus thuirt mo sheanmhar: “Ma tha sibh dha-rìribh airson a dhèanamh, feumaidh sibh tòiseachadh a-nis.” Cha robh duine againn cinnteach dè cho doirbh 's a bhiodh an obair, ach bha sinn deònach feuchainn."
#     "Mu dheireadh, chaidh sinn a-mach don ghàrradh. Bha an t-sìde fuar agus fliuch, agus cha robh mòran solais ann, ach bha seòrsa de shìth anns an àite. Bha mi a' smaoineachadh nach robh àite sam bith eile ann far am b' fheàrr leam a bhith."
# ]
# for t in texts:
#     doc = nlp(t)
#     print("\n", t)
#     for token in doc:
#         print(token.text, token.pos_, token.tag_)


# testing raw arcosg sentences (using limiting-> [:1] to only print the first file)
for file_path in sorted(clean_folder.rglob("*.txt"))[:1]:
    print(f"\n{'=' * 60}")
    print(f"File: {file_path.name}")
    print("=" * 60)

    text = file_path.read_text(encoding="utf-8")

    doc = nlp(text)

    for token in doc:
        print(token.text, token.pos_, token.tag_)


# i=0
# for label in nlp.get_pipe("tagger").labels:
#     print(label)
#     i+=1

# print(i)


# # Load training data
# docbin = DocBin().from_disk("../data/processed_small_tagset/train.spacy")
# docs = list(docbin.get_docs(nlp.vocab))

# total_tokens = sum(len(doc) for doc in docs)

# vocab = set()

# for doc in docs:
#     for token in doc:
#         vocab.add(token.text)

# print("Number of training tokens:", total_tokens)
# print("Vocab size:", len(vocab))