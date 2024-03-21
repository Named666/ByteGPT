import torch
import os
import math
import time
from adamp import AdamP

# Configurations
config = {
    'model_name': 'model.pt', # Name of the model file
    'seed': 666, # The number of the beast
    'batch_size': 8, # how many independent sequences will we process in parallel?
    'block_size': 1024, # what is the maximum context length for predictions?
    'max_iters': 10, # how many training iterations to run?
    'eval_interval': 1, # how often to print evaluation metrics?
    'learning_rate': 1e-3, # how precise are our steps? (perplexity from a lower learning rate is more accurate)
    'weight_decay': 1e-2, # how much do we "regularize" the training? (prevents overfitting)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # what device are we using?
    'eval_iters': 1, # how many iterations to use for evaluation?
    'n_embd': 256, # how many dimensions in the hidden state?
    'n_head': 4, # how many heads in the multi-head attention?
    'n_layer': 4, # how many layers in the model?
    'dropout': 0.0, # how much dropout to use?
    'vocab_size': 256, # Character level tokenizer based on utf-8 encoding. 256 is the number of unique bytes.
    'folder': r'data/', # Filepath for storing model weights.
    'gradient_accumulation_steps': 2, # how many steps to accumulate gradients for
    'max_grad_norm': 1.0, # max gradient norm
    'temperature': 1, # temperature for sampling, 1.0 means no temperature
    'top_k': 10, # top-k sampling, 0 means no restrictions, 40 means only the 40 most likely tokens are considered
    'top_p': 0.666 # top-p sampling, 0 means no restrictions, 0.95 means only tokens with cumulative probability of 0.95 are considered
}

# Dataset Handling
def get_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def encode(s):
    bytes_list = bytes(s, encoding='utf-8')
    bytes_list = list(bytes_list)
    return bytes_list

def decode(l):
    decoded_bytes = bytes(l).decode('utf-8', errors='replace')
    return decoded_bytes


def prepare_text(text):
    encoded_text = encode(text)
    tensor = torch.tensor(encoded_text, dtype=torch.long)
    return tensor

def get_batch(data, start_index, block_size):
    end_index = start_index + block_size
    x = data[start_index:end_index]
    y = data[start_index+1:end_index+1]
    return x, y

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class Head(torch.nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = torch.nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = torch.nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = torch.nn.Dropout(config['dropout'])

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
        self.proj = torch.nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = torch.nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config['n_embd'], 4 * n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd),
            torch.nn.Dropout(config['dropout']),
        )

    def forward(self, x):
        return self.net(x)

class ChunkedFeedForward(torch.nn.Module):
    def __init__(self, dim, chunks=8):
        super(ChunkedFeedForward, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 4 * dim),
            torch.nn.SiLU(inplace=True),  # Use SiLU (Swish) activation
            torch.nn.Linear(4 * dim, dim),
        )
        self.chunks = chunks

    def forward(self, x):
        # Split the input into chunks
        chunks = torch.chunk(x, self.chunks, dim=1)
        
        # Batch the chunks and apply the network
        batched_chunks = torch.cat(chunks, dim=0)
        processed_batch = self.net(batched_chunks)
        processed_chunks = processed_batch.split(x.size(0), dim=0)
        

        return torch.cat(processed_chunks, dim=1)

class TransformerBlock(torch.nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = torch.nn.LayerNorm(config['n_embd'])
        self.ln2 = torch.nn.LayerNorm(config['n_embd'])
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = ChunkedFeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SmallGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(config['vocab_size'], config['n_embd'])
        self.position_embedding_table = torch.nn.Embedding.from_pretrained(positional_encoding(config['block_size'], config['n_embd']), freeze=True)
        self.blocks = torch.nn.Sequential(*[TransformerBlock(config['n_embd'], config['n_head']) for _ in range(config['n_layer'])])
        self.ln_f = torch.nn.LayerNorm(config['n_embd'])
        self.lm_head = torch.nn.Linear(config['n_embd'], config['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config['device']))
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
    
def train_one_epoch(model, optimizer, train_data, block_size, gradient_accumulation_steps, max_grad_norm):
    model.train()
    total_loss = 0.0
    best_loss = 0.0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_iters'], eta_min=0)
    optimizer.zero_grad()
    start = time.perf_counter()
    for start_index in range(0, len(train_data) - block_size + 1, block_size):
        x, y = get_batch(train_data, start_index, block_size)
        x, y = x.to(config['device']), y.to(config['device'])
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        loss.backward()        
        total_loss += loss.item()
        if (start_index // block_size + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            if best_loss == 0.0:
                best_loss = loss.item()
            elif loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(config['folder'], config['model_name']))
            optimizer.zero_grad()
        # estimate time to completion
        estimate = (time.perf_counter() - start) / (start_index + 1) * (len(train_data) - start_index) / 60
        print(f'Epoch Progress: {start_index / len(train_data) * 100:.2f}% | Last: {loss.item():.4f} | Avg: {total_loss / (start_index // block_size + 1):.4f} | Best loss: {best_loss:.4f} | Train Time: {estimate:.2f} minutes', ' | Lrn Rate: ', scheduler.get_last_lr()[0], end='\r')                                                                                                                                                                                                            
    return total_loss / len(train_data)

def evaluate(model, val_data, block_size):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for start_index in range(0, len(val_data) - block_size + 1, block_size):
            x, y = get_batch(val_data, start_index, block_size)
            x, y = x.to(config['device']), y.to(config['device'])
            _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
            total_loss += loss.item()
            print(f'Evaluating... {start_index / len(val_data) * 100:.2f}% | Last batch loss: {loss.item():.4f} | Average loss: {total_loss / (start_index // block_size + 1):.4f}', end='\r')
    return total_loss / len(val_data)

def calculate_perplexity(loss):
    return math.exp(loss)

def generate(model, context, max_new_tokens, temperature, top_k, top_p):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN or Inf found in logits!")

            # Top-k sampling
            values, indices = torch.topk(logits, top_k)
            top_k_logits = torch.zeros_like(logits).scatter_(1, indices, values)
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            top_k_logits[:, indices_to_remove] = float('-inf')
            
            # Sample the next token
            probs = torch.nn.functional.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)
    return context

def main(config):
    torch.manual_seed(config['seed'])
    data = get_data_from_file(os.path.join(config['folder'], 'tiny_shakespear.txt'))
    train_data = prepare_text(data)
    val_data = prepare_text(data[-int(len(data)*0.2347):])
    
    # Initialize model
    model = SmallGPT()
    optimizer = AdamP(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], betas=(0.9, 0.999))
    best_val_loss = None
    try:
        model.load_state_dict(torch.load(os.path.join(config['folder'], config['model_name'])))
    except:
        print('no model found')
    model = model.to(config['device'])
    # Print number of parameters in millions, storage requirements in GB, , and RAM allocation requirements in GB
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    print(sum(p.numel() for p in model.parameters())*4/1e9, 'GB STORAGE')
    print(sum(p.numel() for p in model.parameters())*4/1e9*config['batch_size'], 'GB RAM')
    # Training loop
    for iter in range(config['max_iters']):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, optimizer, train_data, config['block_size'], config['gradient_accumulation_steps'], config['max_grad_norm'])
        val_loss = evaluate(model, val_data, config['block_size'])
        print(f"step {iter}: train loss {train_loss * config['batch_size']:.4f}, val loss {val_loss * config['batch_size']:.4f}")
        print(f"took {time.perf_counter() - start:.2f} seconds")
        
        # Save model if it's the best one so far, if best_val_loss is None then it's the first iteration
        if iter == 0 or not best_val_loss or val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Best validation loss {best_val_loss * config['batch_size']:.4f}")
            torch.save(model.state_dict(), os.path.join(config['folder'], config['model_name']))
        
        # Generate
        context = prepare_text('The Antichrist is among you.')
        generated = generate(model, context.unsqueeze(0), max_new_tokens=216, temperature=config['temperature'], top_k=config['top_k'], top_p=config['top_p'])
        print(decode(generated[0]))
        
        # Calculate perplexity
        train_perplexity = calculate_perplexity(train_loss)
        val_perplexity = calculate_perplexity(val_loss)
        print(f"step {iter}: train perplexity {train_perplexity:.4f}, val perplexity {val_perplexity:.4f}")

if __name__ == '__main__':
    main(config)
