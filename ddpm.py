import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # # independent sequences will we process in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_inters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ---------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
# Average Loss for stablization
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train','val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X,Y = get_batch(split)
      logits,loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads == nn.ModuleList[Head(head_size) for _ in range(num_heads)]
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    out = torch.cat
class BigramLanguageModel(nn.Module):
  """simple next token generation model"""
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.position_embedding_table = nn.Embedding(block_size,n_embd)
    self.blocks = nn.Sequential(
      Block(n_embd, n_head=4),
      Block(n_embd, n_head =4),
      Block(n_embd, n_head=4),
      nn.LayerNorm
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self,idx, targets=None):
    B,T = idx.shape

    tok_emb = self.token_embedding_tale(idx)#(B,T,C)
    pos_emb = self.position_embedding_table(torch.arrange(T))
    x = tok_emb + pos_emb # (B,T.C)
    x = self.blocks(x) # (B,T.C)

    logits = self.token_embedding_table(tok_emb) #(B,T,vocab_size)
    if targets is None:
      loss = None
    else:
      B,T,C = logits.shape
      # change the shape
      logits = logits.view(B*T,C)
      targets = targets.view(B*T)
      loss = F.cross_entropy (logits, targets) # pytorch wants a T.C,
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      # the last token in that concepts
      logits = logits[:, -1, :] # becomes(B,C)

      probs = F.softmax(logits, dim=-1)
      #given the probabilty, sample from the distribution
      idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
      # appened sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
    return idx

class FeedForward:
  """ MLP Layer"""
  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU,
      nn.Linear(4*n_embd, n_embd),
    )

  def forward(self, x):
    return self.net(x)


class Block(nn.Module):

  def __init__(self, n_embd, n_head):

    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self,x):
    x += self.sa(x)
    x += self.ffwd(x)
    return x



# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# lr is the usual learning rate

"""### Training Loop"""

batch_size = 32
for steps in range(100): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



