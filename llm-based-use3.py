import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from os.path import join, dirname
from dotenv import load_dotenv

# 環境変数の読み込み
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

# 学習モデル情報の取得
model_id = os.environ.get("MODEL_ID")
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

question = os.environ.get("QUESTION")
input= tokenizer(question, return_tensors="pt")
eos_token_id = os.environ.get("EOS_TOKEN_ID")

tokens = model.generate(
    **input,
    max_new_tokens = 30,
    eos_token_id = tokenizer.encode(eos_token_id),
    pad_token_id = tokenizer.pad_token_id,
    do_sample = True,
    num_return_sequences = 5
)

for i in range(5):
    output = tokenizer.decode(tokens[i], skip_special_tokens=True)

print(output)