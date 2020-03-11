import spacy
import re
import numpy as np

class tokenize(object):

    def __init__(self, lang, savetokens):
        self.nlp = spacy.load(lang)
        self.lang = lang
        self.savetokens = savetokens

    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        tokens = [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
        if self.savetokens == 1:
            np.save(self.lang + '_sentences.npy', sentence)
            np.save(self.lang + '_tokens.npy', sentence)

        return tokens
