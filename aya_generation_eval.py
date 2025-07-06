from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import evaluate
import pandas as pd

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

# Testing/evaluation setup
MODEL_NAME = "CohereLabs/aya-expanse-8b" 
QUANTIZE_4BIT = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attn_implementation = "flash_attention_2" #experiment with this
quantization_config = None if not QUANTIZE_4BIT else BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

load_path = f"$SCRATCH/urdu_generation/{MODEL_NAME}_urdu_adapter"
model = AutoModelForCausalLM.from_pretrained(
    load_path,
    quantization_config=quantization_config,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(load_path)
model.load_adapter(load_path)
model.eval()

# Perplexity over the test set
test_data_path = './data/test_urdu.jsonl'
df = pd.read_json(path_or_buf=test_data_path, lines=True)
dataset = Dataset.from_pandas(df)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], return_tensors="pt", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Perplexity calculation using evaluate library
perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(model=model, predictions=dataset["text"], tokenizer=tokenizer, max_length=1024)
print(f"Perplexity: {results['perplexity']}")

# Log-probability calculation and analysis for data visualization, TOKEN LEVEL
for i in range(len(tokenized_dataset)):
    inputs = tokenized_dataset[i]["text"].to(device); print(type(inputs))
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
        logprobs = F.log_softmax(logits, dim=-1)
    for t in range(inputs.size[1] - 1):
        current_token = tokenizer.decode(inputs[0, t])
        next_token_id = inputs[0, t + 1].item()
        next_token_str = tokenizer.decode(next_token_id)
        
        preds = logprobs[0, t]
        topk = torch.topk(preds, k=5)
        topk_tokens = tokenizer.convert_ids_to_tokens(topk.indices.tolist())
        topk_scores = topk.values.tolist()
        
        ground_truth_logprob = preds[next_token_id].item()
        
        print(f"Input token: {current_token}")
        print(f"Next GT token: {next_token_str}")
        print(f"Log prob of GT token: {ground_truth_logprob:.4f}")
        print("Top-5 predictions:")
        for token, score in zip(topk_tokens, topk_scores):
            print(f"  {token:<15} -> logP = {score:.4f}")
        print("=" * 40)
    # print(f"Log-probabilities for example {i}: {logprobs.item()}")
    
#Replicate the above for word-level analysis - TODO



