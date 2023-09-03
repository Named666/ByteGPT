import torch
import os
import time

model_name = 'model.pt'
torch.manual_seed(666) # The number of the beast
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 1024 # what is the maximum context length for predictions?
max_iters = 10 # how many training iterations to run?
eval_interval = 1 # how often to print evaluation metrics?
learning_rate = 1e-3 # how precise are our steps? (perplexity from a lower learning rate is more accurate)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1 # how many iterations to use for evaluation?
n_embd = 256 # how many dimensions in the hidden state?
n_head = 4 # how many heads in the multi-head attention?
n_layer = 4 # how many layers in the model?
dropout = 0.0  # how much dropout to use?
vocab_size = 256 # Character level tokenizer based on utf-8 encoding. Can scale this up to 65,536 for unicode.

folder = 'C:/SAP/SmallGPT/data' # Filepath for storing model weights.
example_text = open('data/kjvdat.txt', 'r', encoding='utf-8').read() # Filepath to training data. 

def encode(s):
    return list(map(ord, s))

def decode(l):
    return ''.join(map(chr, l))

def prepare_text(text):
    encoded_text = encode(text)
    tensor = torch.tensor(encoded_text, dtype=torch.long)
    return tensor

tensor = prepare_text(example_text)
n = int(0.666*len(tensor))
train_data = tensor[:n] 
val_data = tensor[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embd, head_size, bias=False)
        self.query = torch.nn.Linear(n_embd, head_size, bias=False)
        self.value = torch.nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.666
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = torch.nn.Linear(n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(torch.nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.ln2 = torch.nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = torch.nn.Embedding(block_size, n_embd)
        self.blocks = torch.nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    model = BigramLanguageModel()
    try:
        model.load_state_dict(torch.load(os.path.join(folder, model_name)))
        mode = input('train or generate? ')
    except:
        print('no model found')
        mode = 'train'
    print(sum(p.numel() for p in model.parameters())*4/1e9, 'GB')
    m = model.to(device)
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    
    if mode != 'train':
        context_text = input('Enter context: ')
        context = torch.tensor(encode(context_text), device=device).unsqueeze(0)
        print(decode(m.generate(context, max_new_tokens=216)[0]))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        for iter in range(max_iters):
            if iter % eval_interval == 0 or iter == max_iters - 1:
                start = time.perf_counter()
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                print(f"took {time.perf_counter() - start:.2f} seconds")

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        context_text = [ 84, 104, 101,  32,  65, 110, 116, 105,  99, 104, 114, 105, 115, 116, 32, 105, 115,  32,  97, 109, 111, 110, 103,  32, 121, 111, 117,  46]
        context = torch.tensor(context_text, device=device).unsqueeze(0)
        print(decode(m.generate(context, max_new_tokens=216)[0]))
        torch.save(model.state_dict(), os.path.join(folder, model_name))