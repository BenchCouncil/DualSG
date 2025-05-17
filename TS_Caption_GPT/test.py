from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 加载 GPT-2 模型和词表
tokenizer = GPT2Tokenizer.from_pretrained("/root/daye/gpt2")
# model = GPT2LMHeadModel.from_pretrained("/root/daye/gpt2")

# 模型进入评估模式
# model.eval()

# 输入词表的数字（ID）
input_ids = [0, 1, 2]  # 这里替换为你的数字列表

# 使用模型的解码方法
decoded_text = tokenizer.decode(tokenizer.eos_token_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
print(tokenizer.bos_token)
print(tokenizer.eos_token)
if decoded_text == '[EOS]':
    print("fku")
print("Decoded text:", decoded_text)
