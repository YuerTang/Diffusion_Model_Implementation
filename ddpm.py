import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # # independent sequences will we process in parallel
block_size = 256 # maximum context length for predictions
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
eval_iters = 200
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

class Head(nn.Module):
  """ one head of self-attention"""
  
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self. query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    # buffer is value that persistent and will be saved alongside parameters
    self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # input of size (batch, time-step, channels)
    # output of size (batch, time-step, head size)
    B,T,C = x.shape
    k = self.key(x) # (B, T, hs)
    q = self.query(x)  # (B, T, hs)
    # compute "affinities"
    wei = q @ k.transpose(-2, -1)*k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    wei = self.dropout(wei)
    v = self.value(x) # (B, T, hs)
    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
    return out

    #perform the weighted aggregation of the values

class FeedForward(nn.Module):
  """ MLP Layer"""
  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

  
class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)


  def forward(self,idx, targets=None):

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
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size: ]
      logits, loss = self(idx_cond)
      # the last token in that concepts
      logits = logits[:, -1, :] # becomes(B,C)

      probs = F.softmax(logits, dim=-1)
      #given the probabilty, sample from the distribution
      idx_next = torch.multinomial(probs, num_samples = 1) # (B,1)
      # appened sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Use pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
  # So every once in a while we evaluate the loss on train and val sets
  if iter % eval_interval == 0 or iter == max_iters -1:
    losses = estimate_loss()
    print(f"step (iter): train loss {losses['train']:.4f}, val loss{losses['val']:.4f}")
    # This is for sampling a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate context from the model
context = torch.zeros((1,1),dtype = torch.long)
print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))

