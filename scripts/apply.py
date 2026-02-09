import spacy
nlp = spacy.load("gd_core_arcosg_sm")
print(nlp.pipe_names)

doc = nlp("Tha mi a' dol dhan sgoil an-diugh.") # I am going to school today

for token in doc:
    print(token.text, token.pos_, token.tag_)