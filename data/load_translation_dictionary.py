import pickle
token_translation_dictionar = pickle.load(open('data/tokenized_translation_dictionary.p', 'rb'))
print(len(token_translation_dictionar))

# translation_dictionar = pickle.load(open('data/translation_dictionary.p', 'rb'))
# print(len(translation_dictionar))

# new_translation_dict = {}
# for token, translation in zip(token_translation_dictionar, translation_dictionar):
#     if token_translation_dictionar[token] != 0:
#         new_translation_dict[translation] = translation_dictionar[translation]

# print(len(new_translation_dict))

#pickle.dump(new_translation_dict, open('data/translation_dictionary.p', 'wb'))