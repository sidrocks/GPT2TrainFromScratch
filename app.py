import os
import gradio as gr
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
import tiktoken

# Model Architecture (same as training)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
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

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

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
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Load model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig()
model = GPT(config)
model_path = os.path.join("models", "best_model.pt")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
enc = tiktoken.get_encoding('gpt2')

print(f"✅ Model loaded on {device}!")

def generate(prompt: str, max_new_tokens: int = 30, top_k: int = 50, temperature: float = 1.0):
    tokens = enc.encode(prompt)
    max_ctx = config.block_size
    if len(tokens) > max_ctx - 1:
        tokens = tokens[-(max_ctx - 1):]
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            logits = logits[:, -1, :] / max(1e-8, temperature)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_idx, -1, ix)
            x = torch.cat([x, next_token], dim=1)
    out_tokens = x[0].tolist()
    return enc.decode(out_tokens)

# Example prompts for dropdown
example_prompts = [
    "To be, or not to be, that is the question:",
    "O Romeo, Romeo! wherefore art thou Romeo?",
    "Once more unto the breach, dear friends, once more;",
    "All the world's a stage,",
    "The lady doth protest too much, methinks."
]

with gr.Blocks() as demo:
    gr.Markdown("# GPT-2 (124M) Shakespeare Text Generator")
    gr.Markdown(
        "GPT-2 (124M) model trained from scratch on Shakespeare's works. "
        "Start with a prompt and generate Shakespearean-style text!"
    )

    with gr.Row():
        inp = gr.Textbox(lines=3, placeholder="Enter prompt here...", label="Prompt")
        out = gr.Textbox(lines=10, label="Generated Text")

    with gr.Row():
        max_tokens = gr.Slider(1, 200, value=30, step=1, label="Max new tokens")
        topk = gr.Slider(1, 200, value=50, step=1, label="Top-k")
        temp = gr.Slider(0.01, 2.0, value=1.0, step=0.01, label="Temperature")

    with gr.Row():
        example_dropdown = gr.Dropdown(
            choices=example_prompts,
            label="Choose example prompt",
            interactive=True
        )
        clear_btn = gr.Button("Clear output")

    def use_example(prompt):
        return prompt

    def clear_output():
        return ""

    example_dropdown.change(fn=use_example, inputs=example_dropdown, outputs=inp)
    clear_btn.click(fn=clear_output, inputs=[], outputs=out)

    btn = gr.Button("Generate")
    btn.click(fn=generate, inputs=[inp, max_tokens, topk, temp], outputs=out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)