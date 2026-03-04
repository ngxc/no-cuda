import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from transformers import BertTokenizerFast
import torch_directml
import warnings

warnings.filterwarnings("ignore", message=".*DML backend.*")
# 忽略掉那个 Mask 不匹配的警告
warnings.filterwarnings("ignore", message=".*Support for mismatched key_padding_mask.*")


class InferConfig:
    vocab_path = "bert-base-chinese"
    model_path = r"E:\deep-learning\ml\minillm_large\large.pt"

    device = torch_directml.device(1)
    max_len = 512
    temperature = 0.85
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1


cfg = InferConfig()


class MiniLLMBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = dim * 4
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.GLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # 数值加固：确保输入进入 LayerNorm 前没有异常
        ln_x = self.ln1(x)
        # 执行注意力机制
        attn_out, _ = self.attn(ln_x, ln_x, ln_x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            MiniLLMBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, input_ids, attention_mask=None):
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        # 修复 1：将 -inf 替换为大负数，提高 DML 稳定性
        # 修复 2：统一掩码类型为 float
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * -1e10, diagonal=1)

        # 确保 key_padding_mask 也是 float 类型
        k_mask = None
        if attention_mask is not None:
            # DML 对 float 类型的掩码支持通常比 bool 更好
            k_mask = (attention_mask == 0).float() * -1e10

        for blk in self.layers:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=None)  # 这里传None，因为causal_mask通常足够

        x = self.ln(x)
        logits = self.head(x)
        return logits


# ======================
# 加载逻辑
# ======================
tokenizer = BertTokenizerFast.from_pretrained(cfg.vocab_path)
special_tokens = ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
vocab_size = len(tokenizer)

model = MiniLLM(vocab_size, 512, 24, 8, cfg.max_len).to(cfg.device)
state_dict = torch.load(cfg.model_path, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()


# ======================
# 增强版推理函数
# ======================
def generate(text_prompt, max_gen_len=512):
    input_text = f"<|im_start|><|user|>{text_prompt}<|im_end|><|im_start|><|assistant|>"
    input_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([input_ids[-cfg.max_len:]], device=cfg.device)

    with torch.no_grad():
        for _ in range(max_gen_len):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / cfg.temperature

            # 修复 3：数值截断，防止 Logits 爆炸
            next_token_logits = torch.clamp(next_token_logits, -50, 50)

            # 惩罚逻辑
            for token_id in set(input_ids[0].tolist()):
                next_token_logits[token_id] /= cfg.repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=cfg.top_k, top_p=cfg.top_p)

            # 修复 4：Softmax 稳定性
            probs = torch.softmax(filtered_logits, dim=-1)

            # 修复 5：处理概率中的 nan/inf (极端补救)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0001, posinf=1.0)
                probs /= probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.convert_tokens_to_ids("<|im_end|>"):
                break

    return tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True).split("<|assistant|>")[-1].strip()


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -1e10

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -1e10
    return logits


if __name__ == "__main__":
    prompt = "介绍一下《百年孤独》这本书。"

    print(f"{prompt}")
    print(f"Assistant: {generate(prompt)}")