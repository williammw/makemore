# %% 
import torch
import matplotlib.pyplot as plt
%matplotlib inline
words = open('names.txt', 'r').read().splitlines()
words[:10]
len(words)


b = {}
for w in words:
  chs = ['<S>'] + list(w) + ['<E>']
  for ch1, ch2 in zip(chs, chs[1:]):
    bigram = (ch1, ch2)
    b[bigram] = b.get(bigram, 0) + 1
    # print(ch1, ch2)


sorted(b.items(), key = lambda kv : -kv[1])
# %%
a = torch.zeros((3, 5), dtype=torch.int32)
a

a[1, 3] = 1  # [row, column]
a
N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
# itos

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    # print(ix1, ix2)
    N[ix1, ix2] += 1




plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

for i in range(27):
  for j in range(27):
    chstr = itos[i] + itos[j]
    plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
    plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis('off')

N[0]

p = N[0].float()
p /= p.sum()
p

# %%
g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
ix


g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
# ix = torch.multinomial(p, 1, replacement=True, generator=g).item()
# itos[ix]
p

torch.multinomial(p, num_samples=20, replacement=True, generator=g)

# %%
