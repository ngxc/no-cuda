import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPVisionModel, CLIPProcessor
from tqdm import tqdm
from PIL import Image
import torch_directml
import warnings


warnings.filterwarnings("ignore", message=".*DML backend.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Support for mismatched key_padding_mask.*")


DEVICE = torch_directml.device(1)

print(f"当前设备: {DEVICE}")

PLACEHOLDER_TOKEN = "<image>"
NUM_PLACEHOLDER = 32
MODEL_CKPT = r"E:\deep-learning\llm\vlm\gpt2=pro\tiny_minillm_clip_retrain_epoch4.pt"
TEST_IMAGE = r"E:\deep-learning\llm\vlm\test\39.jpg"
MAX_LEN = 256


# ==========================================
# 2. 模型组件定义
# ==========================================
class VisionToTextProj(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, num_layers=4, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.LayerNorm(in_dim if i == 0 else hidden_dim),
                nn.Linear(in_dim if i == 0 else hidden_dim,
                          hidden_dim if i != num_layers - 1 else out_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            if i != len(self.layers) - 1:
                x = x + residual
        return x


class TinyMiniLLMCLIPContrastiveVLM(nn.Module):
    def __init__(self, text_model, vision_model, proj, placeholder_token_id):
        super().__init__()
        self.text_model = text_model
        self.vision_model = vision_model
        self.proj = proj
        self.vis_norm = nn.LayerNorm(vision_model.config.hidden_size).to(DEVICE)

    def encode_image(self, vision_inputs):
        for k in vision_inputs:
            vision_inputs[k] = vision_inputs[k].to(DEVICE)
        with torch.no_grad():
            vis_out = self.vision_model(**vision_inputs)
            vis_feats = self.vis_norm(vis_out.last_hidden_state)
        return self.proj(vis_feats)

    def generate(self, vision_inputs, tokenizer, max_gen_len=256, temperature=0.7, top_p=0.8, repetition_penalty=1.15):
        """
        手动生成循环：避开 transformers 库在 DML 上的 gather 算子 Bug
        """
        # 1. 提取图像特征
        vis_proj = self.encode_image(vision_inputs)  # (1, 257, 768)

        # 2. 构造初始文本 Embeddings
        placeholder_token_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
        prompt_text = f"{PLACEHOLDER_TOKEN * NUM_PLACEHOLDER}\n"
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(DEVICE)

        # 获取基础文本 Embedding
        inputs_embeds = self.text_model.transformer.wte(input_ids)

        # 3. 替换占位符特征 (Image Token Merging)
        patches_per_token = vis_proj.size(1) // NUM_PLACEHOLDER
        placeholder_mask = (input_ids == placeholder_token_id)
        idxs = torch.nonzero(placeholder_mask[0], as_tuple=True)[0]

        for i, idx in enumerate(idxs):
            start = i * patches_per_token
            end = min((i + 1) * patches_per_token, vis_proj.size(1))
            if start < vis_proj.size(1):
                # 将图像特征填入对应的文本槽位
                inputs_embeds[0, idx] = vis_proj[0, start:end].mean(dim=0)

        # 4. 推理生成循环
        generated_ids = []
        seen_tokens = set()  # 用于手动处理重复惩罚

        with torch.no_grad():
            for _ in range(max_gen_len):
                # 获取模型前向输出
                outputs = self.text_model(inputs_embeds=inputs_embeds)
                next_token_logits = outputs.logits[0, -1, :] / temperature

                # 数值加固：防止 DML 计算溢出
                next_token_logits = torch.clamp(next_token_logits, -50, 50)

                # 手动应用重复惩罚 (避开 torch.gather)
                for tid in seen_tokens:
                    if next_token_logits[tid] > 0:
                        next_token_logits[tid] /= repetition_penalty
                    else:
                        next_token_logits[tid] *= repetition_penalty

                # 采样过滤与生成
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_p=top_p)
                probs = torch.softmax(filtered_logits, dim=-1)

                # 处理可能出现的 nan (由于 DML 兼容性产生)
                if torch.isnan(probs).any():
                    probs = torch.nan_to_num(probs, nan=1e-5)
                    probs /= probs.sum()

                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()

                # 记录结果
                if next_token_id == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)
                seen_tokens.add(next_token_id)

                # 更新下一步的 Embeddings
                next_token_embed = self.text_model.transformer.wte(next_token.unsqueeze(0))
                inputs_embeds = torch.cat([inputs_embeds, next_token_embed], dim=1)

                # 显存保护：防止序列无限增长
                if inputs_embeds.size(1) >= 1024:
                    break

        return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ==========================================
# 3. 辅助函数
# ==========================================
def top_k_top_p_filtering(logits, top_p=0.8, filter_value=-1e10):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    # 将掩码右移，确保至少保留一个 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = filter_value
    return logits


# ==========================================
# 4. 主程序：加载与推理
# ==========================================
if __name__ == "__main__":
    print("正在初始化基础组件...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if PLACEHOLDER_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens([PLACEHOLDER_TOKEN])

    vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("加载模型权重")
    text_model = AutoModelForCausalLM.from_pretrained("gpt2")
    text_model.resize_token_embeddings(len(tokenizer))
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    proj = VisionToTextProj(vision_model.config.hidden_size, text_model.config.n_embd)

    # 加载 Checkpoint
    ckpt = torch.load(MODEL_CKPT, map_location="cpu", weights_only=True)
    proj.load_state_dict(ckpt["proj_state_dict"])
    text_model.load_state_dict(ckpt["text_model_state_dict"])

    # 移动到 DML 加速设备
    proj.to(DEVICE)
    text_model.to(DEVICE)
    vision_model.to(DEVICE)

    vlm = TinyMiniLLMCLIPContrastiveVLM(text_model, vision_model, proj,
                                         tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN))
    vlm.eval()

    # 执行推理
    image_path = TEST_IMAGE
    if not os.path.exists(image_path):
        print(f"找不到测试图片: {image_path}")
    else:
        print("正在处理图片并进行推理...")
        image = Image.open(image_path).convert("RGB")
        v_inputs = vision_processor(images=image, return_tensors="pt")

        print("\n" + "=" * 40)
        print(" VLM 生成结果:")
        print("-" * 40)
        # 调用我们自己写的 generate 逻辑
        response = vlm.generate(v_inputs, tokenizer, max_gen_len=MAX_LEN)
        print(response)
        print("=" * 40)