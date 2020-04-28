from nltk.translate.bleu_score import sentence_bleu

fr_test = open('data/fr_test').read().strip().split('\n')
fr_test_preds = open('data/fr_test_translations_informed').read().strip().split('\n')

reference = [x.replace('.', '').lower().split(' ') for x in fr_test]
candidate = [x.replace('.', '').lower().split(' ') for x in fr_test_preds]

BLEU = 0 
for i in range(len(reference)):
  BLEU += sentence_bleu([reference[i]], candidate[i], weights=(0.5, 0.5))
BLEU /= (i+1)

print('BLEU score: ', BLEU)