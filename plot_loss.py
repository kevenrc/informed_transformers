import torch
import matplotlib.pyplot as plt 

data = torch.load('../final results/small wo alignment/loss_history.pt')#[::100]
data2 = torch.load('../final results/small w alignment/loss_history.pt')#[::100]

plt.title('Vanilla Transformer')
plt.plot(data[:, 2], label='Lexical loss')
plt.plot(data[:, 1], label='Alignment loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.ylim(0, 7)
plt.legend()

plt.figure()
plt.title('Informed Transformer')
plt.plot(data2[:, 2], label='Lexical loss')
plt.plot(data2[:, 1], label='Alignment loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.ylim(0, 7)
plt.legend()

plt.figure()
plt.plot(data[:, 2], label='Lexical loss (Vanilla Transformer)', alpha=0.6)
plt.plot(data2[:, 2], label='Lexical loss (Informed Transformer)', alpha=0.6)
plt.legend()
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
# plt.ylim(0, 0.3)


print(min(data[:, 2]), min(data2[:, 2]))

plt.show()