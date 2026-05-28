import spacy
#nlp = spacy.load("gd_core_arcosg_sm")
nlp = spacy.load("../training/output_coarse/model-best")
print(nlp.pipe_names)

doc = nlp("Tha mi a' dol dhan sgoil an-diugh.") # I am going to school today

for token in doc:
    print(token.text, token.pos_, token.tag_)

texts = [
"Tha mi toilichte an-diugh.",
"Chuala mi sgeulachd bho sheann duine anns a' bhaile.",
"Bha an cù beag a' ruith tron phàirc.",
"Thèid sinn dhan sgoil a-màireach.",
"Carson a tha i cho brònach?",
"Chunnaic Iain an càr dearg aig an taigh mhòr.",
"Chan eil mi a' tuigsinn an leasan seo.",
"Bha na daoine a' bruidhinn gu luath agus gu sunndach.",
"Tha Alba na dùthaich bhrèagha.",
"Am faca tu am film ùr a-raoir?",
"Ghabh e biadh anns a' chidsin mhòr.",
"Bha e glè fhuar ach bha a' ghrian a' deàrrsadh.",
"Thàinig iad dhachaigh an dèidh na h-obrach.",
"Tha mi a' smaoineachadh gu bheil e ceart.",
"Càite an deach an leabhar agam?"
]

for t in texts:
    doc = nlp(t)
    print("\n", t)
    for token in doc:
        print(token.text, token.pos_, token.tag_)