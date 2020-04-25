import torch
translation_dictionar = {'i':'je', 'animal':'animal'}
dictionary = translation_dictionar
y_t = 'je'
y_t = 'animal'

input_sequence = ['i', 'like', 'an', 'animal']

def get_one_hot_vectors(input_sequence, y_t):
  one_hot_vector = torch.zeros(len(input_sequence))
  for i, en_word in enumerate(input_sequence):
      if dictionary.get(en_word, None) == y_t:
          one_hot_vector[i] = 1
  return one_hot_vector