# Solving for residual std scaling issue
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint # Moved this import to the top

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    # In forward of Block:
    def forward(self, x):
        def _forward_block(x):
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x
        return checkpoint(_forward_block, x)


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 6 # number of layers (reduced from 12)
    n_head: int = 6 # number of heads (reduced from 12)
    n_embd: int = 384 # embedding dimension (reduced from 768)


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)



    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    # Device setup same as before
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Hyperparameters
B, T = 8,128  # batch size and sequence length (8192 tokens per batch)
max_iters = 2000
warmup_iters = 200
base_lr = 3e-4
final_lr = 1e-5
grad_clip = 1.0
patience = 20  # early stopping patience
num_val_batches = 10
accum_steps = 4  # effectively batch 32 by accumulation

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
# Load full tokens
with open('input.txt', 'r') as f:
    text = f.read()
enc = tiktoken.get_encoding('gpt2')
tokens = torch.tensor(enc.encode(text))

# Simple 90/10 train-val split to avoid data leakage
num_train_tokens = int(0.9 * len(tokens))
train_tokens = tokens[:num_train_tokens]
val_tokens = tokens[num_train_tokens:]

# Create data loaders pointing to split tokens
train_loader = DataLoaderLite(B, T)
train_loader.tokens = train_tokens
train_loader.current_position = 0

val_loader = DataLoaderLite(B, T)
val_loader.tokens = val_tokens
val_loader.current_position = 0

# Clear CUDA cache before model initialization
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Initialize model
model = GPT(GPTConfig())
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

# Learning rate schedule: linear warmup + cosine decay
def get_lr(step):
    if step < warmup_iters:
        return base_lr * step / warmup_iters
    progress = (step - warmup_iters) / (max_iters - warmup_iters)
    return final_lr + 0.5 * (base_lr - final_lr) * (1 + math.cos(math.pi * progress))

best_val_loss = float('inf')
no_improve_steps = 0
train_losses = [] # Initialize train_losses list

model.train()

from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda')

for step in range(max_iters):
    optimizer.zero_grad()
    for _ in range(accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with autocast('cuda'):
            logits, loss = model(x, y)
            loss = loss / accum_steps  # scale loss

        scaler.scale(loss).backward()

    # Gradient clipping and optimizer step
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.empty_cache()
    train_losses.append(loss.item()) # Append training loss after each step

  # Validation and logs every N steps (adjust for accum steps)
    if step % (100 // accum_steps) == 0 or step == max_iters - 1:
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(num_val_batches):
                xv, yv = val_loader.next_batch()
                xv, yv = xv.to(device), yv.to(device)
                _, val_loss = model(xv, yv)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        lr = get_lr(step) # Assign the learning rate
        print(f"Step {step}: train loss {train_losses[-1]:.5f}, val loss {avg_val_loss:.5f}, lr {lr:.6f}") # Use train_losses[-1]

        # Early stopping and checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_steps = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("Checkpoint saved.")
        else:
            no_improve_steps += 1
            if no_improve_steps >= patience:
                print("Early stopping triggered.")
                break
        model.train()

# Load best model for sampling/generation
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

# Sampling/generation code (unchanged from original)
num_return_sequences = 5
max_length = 30
x = val_loader.next_batch()[0][:num_return_sequences].to(device)  # start from some validation tokens

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[0]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)