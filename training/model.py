"""
LLaMA-Style Decoder-Only Transformer for Tool-Augmented Report Generation.

Architecture upgrades from the original GPT-style model:
  - RoPE (Rotary Positional Encoding) instead of sinusoidal
  - RMSNorm instead of LayerNorm (faster, more stable on MPS)
  - SwiGLU FFN instead of GELU (proven ~10% better perplexity)
  - Tool-aware special tokens for structured tool output
  - 2048 context window (up from 1024)
  - Cross-platform: MPS / CUDA / CPU

Target: ~124M parameters. Pure PyTorch, no pretrained weights.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Tool-Aware Special Tokens
# =============================================================================

SPECIAL_TOKENS = [
    "<|tool_start|>",
    "<|tool_end|>",
    "<|search_result|>",
    "<|scrape_result|>",
    "<|source|>",
    "<|url|>",
    "<|report_start|>",
    "<|report_end|>",
]


def add_special_tokens(tokenizer):
    """Add tool-aware special tokens to the tokenizer and return the count added."""
    new_tokens = {"additional_special_tokens": SPECIAL_TOKENS}
    num_added = tokenizer.add_special_tokens(new_tokens)
    return num_added


def get_special_token_ids(tokenizer) -> dict:
    """Return a mapping of special token name -> token id."""
    return {tok: tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS}


# =============================================================================
# Cross-Platform Device Utilities
# =============================================================================

def get_device(preference: str = "auto") -> torch.device:
    """Dynamic device router: CUDA > MPS > CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_training_dtype(device: torch.device) -> torch.dtype:
    """Device-aware dtype: bfloat16 on CUDA, float32 on MPS/CPU.

    MPS float16 autocast is handled by torch.autocast at the call site;
    storing weights in float32 avoids MPS half-precision accumulation bugs.
    """
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    """Dtype for torch.autocast context manager."""
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


# =============================================================================
# Configuration
# =============================================================================

class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 2048,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_token_id: int = 50256,
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        rope_base: float = 10000.0,
        num_special_tokens: int = len(SPECIAL_TOKENS),
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.rope_base = rope_base
        self.num_special_tokens = num_special_tokens

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads

    @property
    def effective_vocab_size(self) -> int:
        return self.vocab_size + self.num_special_tokens

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        valid_keys = {
            'vocab_size', 'max_seq_len', 'd_model', 'n_heads', 'n_layers',
            'd_ff', 'dropout', 'pad_token_id', 'eos_token_id', 'bos_token_id',
            'rope_base', 'num_special_tokens',
        }
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# RMSNorm (replaces LayerNorm — faster, more stable on MPS)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# Rotary Positional Encoding (RoPE)
# =============================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to input tensor x of shape (batch, n_heads, seq, d_head)."""
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]

    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, d_half)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)
    return rotated


# =============================================================================
# Multi-Head Self-Attention with RoPE
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        output = self.resid_dropout(output)
        return output


# =============================================================================
# SwiGLU Feed-Forward Network
# =============================================================================

class SwiGLUFeedForward(nn.Module):
    """SwiGLU(x) = (x @ W_gate * silu(x @ W_up)) @ W_down"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = self.w_down(gate * up)
        x = self.dropout(x)
        return x


# =============================================================================
# Transformer Block (Pre-RMSNorm + RoPE Attention + SwiGLU)
# =============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, d_head, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ff = SwiGLUFeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


# =============================================================================
# Full Decoder-Only Transformer Model
# =============================================================================

class ScratchTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.effective_vocab_size, config.d_model)

        self.rotary_emb = RotaryEmbedding(
            config.d_head, config.max_seq_len, config.rope_base
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, config.n_heads, config.d_head,
                config.d_ff, config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        self.ln_final = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.effective_vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {n_params / 1e6:.1f}M parameters")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                torch.nn.init.ones_(module.weight)

        for block in self.blocks:
            torch.nn.init.normal_(
                block.attn.W_o.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers),
            )
            torch.nn.init.normal_(
                block.ff.w_down.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers),
            )

    def resize_token_embeddings(self, new_vocab_size: int, tokenizer=None):
        """Resize embeddings for new special tokens, warm-starting from existing."""
        old_size = self.token_embedding.num_embeddings
        if new_vocab_size == old_size:
            return

        old_weight = self.token_embedding.weight.data
        self.token_embedding = nn.Embedding(new_vocab_size, self.config.d_model)
        self.token_embedding.weight.data[:old_size] = old_weight

        mean_emb = old_weight.mean(dim=0)
        for i in range(old_size, new_vocab_size):
            self.token_embedding.weight.data[i] = mean_emb

        self.lm_head = nn.Linear(self.config.d_model, new_vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.config.vocab_size = new_vocab_size - self.config.num_special_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len, (
            f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
        )

        x = self.token_embedding(input_ids)

        cos, sin = self.rotary_emb(seq_len)

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        result = {'logits': logits}

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 300,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        eos_token_id: int = None,
    ) -> torch.Tensor:
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            context = generated[:, -self.config.max_seq_len:]
            output = self.forward(context)
            logits = output['logits'][:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            logits = logits / temperature

            if top_k > 0:
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                logits = logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

        return generated


# =============================================================================
# Helper: Count parameters
# =============================================================================

def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_millions': total / 1e6,
        'trainable_millions': trainable / 1e6,
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing ScratchTransformer (LLaMA-style)")
    print("=" * 60)

    config = TransformerConfig()
    model = ScratchTransformer(config)

    params = count_parameters(model)
    print(f"\nTotal parameters: {params['total_millions']:.1f}M")
    print(f"Trainable parameters: {params['trainable_millions']:.1f}M")
    print(f"\nConfig: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, d_ff={config.d_ff}")
    print(f"Max sequence length: {config.max_seq_len}")
    print(f"Effective vocabulary size: {config.effective_vocab_size}")

    print("\n--- Forward Pass Test ---")
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.effective_vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    output = model(input_ids, labels=labels)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")

    print("\n--- Generation Test ---")
    prompt = torch.randint(0, config.effective_vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Prompt length: 10, Generated length: {generated.shape[1]}")

    print("\nAll tests passed!")
