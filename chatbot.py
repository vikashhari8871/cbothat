import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure that necessary resources are downloaded (run this once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize the sentence
tokens = word_tokenize(sentence)

# Perform POS tagging
pos_tags = pos_tag(tokens)

# Print the sentence with its POS tags
print("Sentence with POS tags:")
for word, tag in pos_tags:
    print(f"{word}: {tag}")
