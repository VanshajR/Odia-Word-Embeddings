from gensim.models import Word2Vec
from indic_transliteration.sanscript import transliterate, ORIYA, ITRANS
from googletrans import Translator

# Path to your trained model
model_path = "models/word2vec_sg.model"

# Load the model
model = Word2Vec.load(model_path)
print("Model loaded!")

# Print vocabulary size
vocab_size = len(model.wv)
print(f"Vocabulary size: {vocab_size}")

translator = Translator()

# Test words (edit as needed)
# test_words = ["ଉତ୍ତମ ", "ଭାଷା", "ଭାରତ"]
test_words = [
    "ଭଲ",     # good
    "ମନ୍ଦ",     # bad
    "ପ୍ରେମ",   # love
    "ଘୃଣା",    # hate
    "ଶିକ୍ଷା",   # education
    "ଗ୍ରାମ",   # village
    "ନଗର",     # city
    "ପାଣି",     # water
    "ଅଗ୍ନି",    # fire
    "ବାୟୁ"     # air
]

for word in test_words:
    if word in model.wv:
        translit = transliterate(word, ORIYA, ITRANS)
        try:
            translation = translator.translate(word, src='or', dest='en').text
        except Exception as e:
            translation = f"(translation error: {e})"
        print(f"\nWords similar to '{word}' ({translit}, {translation}):")
        for sim_word, score in model.wv.most_similar(word, topn=5):
            sim_translit = transliterate(sim_word, ORIYA, ITRANS)
            try:
                sim_translation = translator.translate(sim_word, src='or', dest='en').text
            except Exception as e:
                sim_translation = f"(translation error: {e})"
            print(f"  {sim_word} ({sim_translit}, {sim_translation}): {score:.4f}")
    else:
        print(f"\n'{word}' not in vocabulary.") 