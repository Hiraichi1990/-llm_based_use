import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/open-calm-small"
)

tokenizer = AutoTokenizer.from_pretrained(
    "cyberagent/open-calm-small"
)