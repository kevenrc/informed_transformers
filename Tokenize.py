import spacy
import re
import numpy as np

class tokenize(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
            
    def tokenizer(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        tokens = [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
        np.save(lang + '_sentences.npy', sentence)
        np.save(lang + '_tokens.npy', sentence)
        
        return tokens
