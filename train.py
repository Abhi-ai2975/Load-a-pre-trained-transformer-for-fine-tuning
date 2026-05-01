import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Define Model and Dataset
model_id = "Qwen/Qwen2.5-0.5B" 
dataset_name = "yahma/alpaca-cleaned" 

print(f"Loading {model_id} in 4-bit...")

# 2. Configure 4-bit Quantization (Saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 3. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto" 
)

# 4. Prepare for LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 5. Load and Format Dataset
print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train[:1000]")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"Instruction: {example['instruction'][i]}\nInput: {example['input'][i]}\nOutput: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

# 6. Setup Trainer
print("Starting training...")
training_args = TrainingArguments(
    output_dir="./qwen-finetuned",
    per_device_train_batch_size=1, # Keep at 1 for 6GB VRAM
    gradient_accumulation_steps=4, # Simulates a batch size of 4
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_steps=100, 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=256, 
    args=training_args,
)

# 7. Execute Training Loop
trainer.train()

# 8. Save the newly trained adapters
trainer.model.save_pretrained("qwen-finetuned-adapter")
print("Training complete! Adapter saved.")
