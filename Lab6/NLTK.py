import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag
from nltk import CFG, ChartParser
import string

# Ensure you have the required packages installed.
# pip install nltk


# auto downloads required NLTK data in code run.
nltk_packages = [
    'punkt', 
    'punkt_tab', 
    'averaged_perceptron_tagger', 
    'averaged_perceptron_tagger_eng'
]
for pkg in nltk_packages:
    nltk.download(pkg)

# The English sentence.
text = "The cat chased the mouse."

# 1. Lexical Analysis
print("\n1. Lexical Analysis")
tokens = word_tokenize(text)
print("Tokens:", tokens)

stemmer = PorterStemmer()
stems = [stemmer.stem(w) for w in tokens]
print("Stems:", stems)

pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)

tokens_no_punct = [t for t in tokens if t not in string.punctuation]

# 2. Syntactic Analysis
print("\n2. Syntactic Analysis")
grammar = CFG.fromstring("""
S -> NP VP
NP -> Det N
VP -> V NP
Det -> 'The' | 'the'
N -> 'cat' | 'mouse'
V -> 'chased'
""")
parser = ChartParser(grammar)

parsed = False
for tree in parser.parse(tokens_no_punct):
    print(tree)
    parsed = True
if not parsed:
    print("No parse tree found. Check grammar or input tokens.")

# 3. Semantic Analysis
print("\n3. Semantic Analysis")
semantics = {('cat', 'mouse', 'chased'): "chase(cat, mouse)"}
predicate = semantics.get(('cat', 'mouse', 'chased'))
print("Predicate Logic:", predicate)

# 4. Pragmatic Analysis
print("\n4. Pragmatic Analysis")
context = {"The cat": "Fluffy the pet cat", "the mouse": "Jerry the cartoon mouse"}
resolved = f"{context['The cat']} chased {context['the mouse']}"
print("Contextual meaning:", resolved)

# 5. Machine Translation (English → Spanish)
print("\n5. Machine Translation (English → Spanish)")
eng_to_spa = {"The": "El", "the": "el", "cat": "gato", "mouse": "ratón", "chased": "persiguió"}
translation = [eng_to_spa.get(w, w) for w in tokens]
print(" ".join(translation))
