
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
#from deepseekmodel import DeepSeekModel
from model import DeepSeekLM
import os
import time

print(f"PyTorch Version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 1. Configuration
model_id = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = AutoConfig.from_pretrained(model_id)
config.vocab_size = len(tokenizer)

# DeepSeek Config
config.q_lora_rank = 128
config.kv_lora_rank = 128
config.nope_head_dim = 32
config.rope_head_dim = 32
config.num_shared_experts = 2
config.num_routed_experts = 8
config.num_active_experts = 2
config.expert_intermediate_size = 384

# 2. Model
model = DeepSeekLM(config).to(device)
print("DeepSeek Model Initialized.")

data_file = "input-1.txt"
if os.path.exists(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        full_text = f.read()
    print(f"Loaded local data from {data_file}")
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": [full_text]})
else:
    print(f"Data file {data_file} not found. Downloading default dataset from Hugging Face...")
    from datasets import load_dataset
    # Use a small subset of a public dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train[:1%]")
    print("Loaded default dataset from Hugging Face.")



block_size = 256

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
lm_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000)
lm_dataset = lm_dataset.with_format("torch")

train_dataloader = DataLoader(lm_dataset, batch_size=16, shuffle=True, pin_memory=True)
print(f"Dataset prepared. Number of chunks: {len(lm_dataset)}. Batch size: 16")

# 4. Training Setup
# Optimization: Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

def generate_text(model, tokenizer, prompt="The meaning of life is", max_new_tokens=50):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model(input_ids)
                logits = outputs if not isinstance(outputs, tuple) else outputs[0]
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    print(f"Generated: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
    model.train()

# 5. Training Loop
steps = 0
max_steps = 10000
save_path = "deepseek_checkpoint_10000.pt"
accumulation_steps = 4
start_time = time.time()
total_tokens = 0

model.train()
print("Starting training...")

while steps < max_steps:
    for batch in train_dataloader:
        if steps >= max_steps:
            break
            
        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()
        
        if steps % accumulation_steps == 0:
            optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            logits, expert_usage = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            loss = loss / accumulation_steps
            
            # Accuracy
            with torch.no_grad():
                preds = torch.argmax(shift_logits, dim=-1)
                correct = (preds == shift_labels).sum()
                accuracy = correct.float() / shift_labels.numel()

        scaler.scale(loss).backward()
        
        if (steps + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
        
        steps += 1
        total_tokens += input_ids.numel()
        
        if steps % 100 == 0:
            elapsed = time.time() - start_time
            tps = total_tokens / elapsed
            print(f"Step {steps}: Loss {loss.item() * accumulation_steps:.4f} | Acc {accuracy.item():.4f} | TPS {tps:.2f}")
            if expert_usage is not None:
                top = torch.topk(expert_usage, k=3)
                print(f"  Top Experts: {top.indices.tolist()} (Counts: {top.values.tolist()})")

        if steps % 1000 == 0:
            print(f"\n--- Step {steps} Generation ---")
            generate_text(model, tokenizer)
            print("-----------------------------\n")

torch.save(model.state_dict(), save_path)
print(f"Checkpoint saved to {save_path}")

print("\n--- Final Generations (5 Outputs) ---")
prompts = ["The future of AI is", "Once upon a time", "In a galaxy far away", "The secret to happiness is", "Python is a programming language that"]
for i, p in enumerate(prompts):
    print(f"\nGeneration {i+1}:")
    generate_text(model, tokenizer, prompt=p, max_new_tokens=100)
print("-------------------------------------")
