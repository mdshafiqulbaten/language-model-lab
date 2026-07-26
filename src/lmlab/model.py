from __future__ import annotations


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Install the model extra: pip install -e '.[model]'") from exc
    return torch, nn, functional


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalSelfAttention(nn.Module):
        def __init__(self, width: int, heads: int, dropout: float = 0.0):
            super().__init__()
            if width % heads:
                raise ValueError("width must be divisible by heads")
            self.heads, self.dim, self.dropout = heads, width // heads, dropout
            self.qkv = nn.Linear(width, 3 * width)
            self.out = nn.Linear(width, width)

        def forward(self, x):
            batch, length, width = x.shape
            query, key, value = self.qkv(x).chunk(3, dim=-1)

            def split(tensor):
                return tensor.view(
                    batch, length, self.heads, self.dim
                ).transpose(1, 2)

            query, key, value = map(split, (query, key, value))
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
            output = output.transpose(1, 2).contiguous().view(batch, length, width)
            return self.out(output)

    class Block(nn.Module):
        def __init__(self, width: int, heads: int, expansion: int = 4):
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.attention = CausalSelfAttention(width, heads)
            self.norm2 = nn.LayerNorm(width)
            self.feed_forward = nn.Sequential(
                nn.Linear(width, expansion * width),
                nn.GELU(),
                nn.Linear(expansion * width, width),
            )

        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            return x + self.feed_forward(self.norm2(x))

    class TinyGPT(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            context_length: int,
            width: int = 128,
            layers: int = 4,
            heads: int = 4,
        ):
            super().__init__()
            if min(vocab_size, context_length, width, layers, heads) < 1:
                raise ValueError("model dimensions must be positive")
            self.context_length = context_length
            self.token_embedding = nn.Embedding(vocab_size, width)
            self.position_embedding = nn.Embedding(context_length, width)
            self.blocks = nn.ModuleList(
                [Block(width, heads) for _ in range(layers)]
            )
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, vocab_size, bias=False)
            self.head.weight = self.token_embedding.weight

        def forward(self, token_ids, targets=None):
            batch, length = token_ids.shape
            if length > self.context_length:
                raise ValueError("sequence exceeds context length")
            positions = torch.arange(length, device=token_ids.device)
            x = self.token_embedding(token_ids) + self.position_embedding(positions)
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.norm(x))
            loss = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
            return logits, loss

        @torch.no_grad()
        def generate(self, token_ids, max_new_tokens: int = 8):
            for _ in range(max_new_tokens):
                window = token_ids[:, -self.context_length :]
                logits, _ = self(window)
                next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
                token_ids = torch.cat([token_ids, next_id], dim=1)
            return token_ids

except ImportError:
    CausalSelfAttention = Block = TinyGPT = None

