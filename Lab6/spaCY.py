import spacy
import nltk
from nltk.corpus import wordnet as wn

# Ensure you have the required packages installed.
# pip install spacy nltk
# python -m spacy download en_core_web_sm

#this auto installs in code run.
nltk.download('wordnet')
nltk.download('omw-1.4')

# English sentence
text = "Apple is looking at buying a startup in Silicon Valley."

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# 1. Lexical Analysis
print("\n1. Lexical Analysis")
print("Tokens:", [token.text for token in doc])
print("Lemmas:", [token.lemma_ for token in doc])
print("Morphology:", [token.morph for token in doc])

# 2. Syntactic Analysis
print("\n2. Syntactic Analysis")
print("POS Tags:", [(token.text, token.pos_) for token in doc])
print("Dependency Parse:")
for token in doc:
    print(f"{token.text} --> {token.dep_} --> {token.head.text}")

# 3. Semantic Analysis
print("\n3. Semantic Analysis")

# Example: similarity between first two nouns
nouns = [token for token in doc if token.pos_ == "NOUN"]
if len(nouns) >= 2:
    print(f"Similarity ({nouns[0].text}, {nouns[1].text}):", nouns[0].similarity(nouns[1]))
    
# Synonyms via WordNet
word = "startup"
syns = wn.synsets(word)
print(f"Synonyms for '{word}':", set(lemma.name() for s in syns for lemma in s.lemmas()))


# 4. Pragmatic Analysis
print("\n=== 4. Pragmatic Analysis ===")
print("Named Entities:", [(ent.text, ent.label_) for ent in doc.ents])

# Simple context understanding: check if the text mentions a company
if any(ent.label_ == "ORG" for ent in doc.ents):
    print("Context: Text mentions an organization/company.")
else:
    print("Context: No organization mentioned.")


# 5. Machine Translation (English → Spanish)
print("\n5. Machine Translation (Demo)")
eng_to_spa = {
    "Apple": "Apple",
    "is": "está",
    "looking": "buscando",
    "at": "en",
    "buying": "comprar",
    "a": "una",
    "startup": "startup",
    "in": "en",
    "Silicon": "Silicon",
    "Valley": "Valle"
}
translation = [eng_to_spa.get(token.text, token.text) for token in doc]
print(" ".join(translation))
