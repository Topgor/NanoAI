import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class SparseExpert(nn.Module):
    def __init__(self, dim, hidden_mult=3):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MixtureOfExperts(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SparseExpert(dim) for _ in range(num_experts)])
    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.gate(x_flat)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            idx = indices[:, i]
            w = weights[:, i].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = (idx == e)
                if mask.any():
                    output[mask] += w[mask] * self.experts[e](x_flat[mask])
        return output.view(B, S, D)

class SparseAttention(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
    def forward(self, x, mask=None):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rope(x)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        causal = torch.tril(torch.ones(S, S, device=x.device)).bool()
        attn = attn.masked_fill(~causal, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, S, D))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=8, n_kv_heads=2, use_moe=True):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = SparseAttention(dim, n_heads, n_kv_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.ffn = MixtureOfExperts(dim) if use_moe else SparseExpert(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class NanoAI(nn.Module):
    def __init__(self, vocab_size=32000, dim=256, n_layers=8, n_heads=8, n_kv_heads=2, max_seq=256):
        super().__init__()
        self.dim = dim
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, use_moe=(i % 2 == 0))
            for i in range(n_layers)
        ])
        self.norm = nn.RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.tok_emb.weight
        self._init_weights()
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x)
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss
    def count_params(self):
        t = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return t, tr
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=-1)
        return idx
    def save(self, path):
        torch.save({'model_state': self.state_dict(), 'config': {'vocab_size': self.vocab_size, 'dim': self.dim, 'max_seq': self.max_seq}}, path)
        print(f"Saved: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)")

if __name__ == "__main__":
    print("NANO-AI: Init...")
    model = NanoAI()
    t, tr = model.count_params()
    print(f"Params: {t:,} (~{t*4/1024/1024:.1f} MB)")
    test = torch.randint(0, 1000, (1, 10))
    logits, _ = model(test)
    print(f"Forward OK! Shape: {logits.shape}")
    print("ARCHITECTURE READY!")
