from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import pandas as pd

'''Script for finetuning models on custom Urdu dataset for next-token generation.
Specify the model via HuggingFace and the script uses the SFTTrainer and trl library
for model finetuning. We make use of the following optimizations:
- LoRA (and QLoRA if choosing a quantized model configuration)
- FlashAttention
- gradient accumulation

Structure is adapted from Cohere's AYA Expanse SFT example with added code to
handle other models.

The goals of the generation finetuning task are as follows:
- investigate improvement in model next-token prediction performance in Urdu
- characterize changes in model internal structure and representation space
   as a result of finetuning
   '''
   
# Hyperparameter setup
QUANTIZE_4BIT = True
TRAIN_BATCH_SIZE = 16
TRAIN_MAX_SEQ_LENGTH = 512
USE_FLASH_ATTENTION = True
GRAD_ACC_STEPS = 2

MODEL_NAME = "meta-llama/Llama-3.2-3B"  # Experiment with an 8B non-multilingual model as well

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attn_implementation = "flash_attention_2" #experiment with this
quantization_config = None if not QUANTIZE_4BIT else BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

train_data_path = './data/training_urdu.jsonl'
valid_data_path = './data/validation_urdu.jsonl'
test_data_path = './data/test_urdu.jsonl'

# Dataset setup
df = pd.read_json(path_or_buf=train_data_path, lines=True)
dataset = Dataset.from_pandas(df)
valid_df = pd.read_json(path_or_buf=valid_data_path, lines=True)
val_dataset = Dataset.from_pandas(valid_df)
#the datasets here are setup as trl "Standard" datasets
#they expect a dictionary for each example with a "text" key

# Add a check to ensure that the data is formatted correctly.


model = AutoModelForCausalLM.from_pretrained(
          MODEL_NAME,
          quantization_config=quantization_config,
          attn_implementation=attn_implementation,
          torch_dtype=torch.bfloat16,
        )
model = model.to(device)
#model is loaded in fp 16 -- this is 2 bytes per param

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#might want to experiment with adjusting the tokenizer for Urdu-specific down the line

# For config setup, we will aim to run through about 1000 examples.
# With a batch size of 16, this will need about 64 steps per epoch.
# We will set 25 training epochs, which will give us about 1600 steps.
   
# SFT Language Model Training Config
training_arguments = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=25,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC_STEPS,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=1e-3,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"
)
llama_peft_config = LoraConfig(
    lora_alpha=16,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=val_dataset,
    peft_config=llama_peft_config,
    tokenizer=tokenizer,
    args=training_arguments
)

# Train the model
trainer.train()

# Write final model to disk
save_path = f"$SCRATCH/urdu_generation/{MODEL_NAME}_urdu_adapter"
trainer.model.save_pretrained(save_path)
model.config.use_cache = True
model.eval()
