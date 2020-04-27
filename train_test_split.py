import numpy as np
en = open('data/small_vocab_en').read().strip().split('\n')
fr = open('data/small_vocab_fr').read().strip().split('\n')
ids = np.arange(0, 137860)
split = int(0.85*len(ids))
np.random.shuffle(ids)

ids[:10]

en_train, en_test = en[:split], en[split:]
fr_train, fr_test = fr[:split], fr[split:]

print(en_train[100], fr_train[100])
print(en_test[100], fr_test[100])

open('data/en_train', 'w').write('\n'.join(en_train))
open('data/en_test', 'w').write('\n'.join(en_test))
open('data/fr_train', 'w').write('\n'.join(fr_train))
open('data/fr_test', 'w').write('\n'.join(fr_test))

print(len(en), len(fr), len(ids), len(en_train), len(en_test), len(fr_train), len(fr_test))