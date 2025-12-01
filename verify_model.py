import torch
from model import DeepSeekLM, DeepSeekMLA, DeepSeekMoE
from transformers import AutoConfig

# Mock Config
class MockConfig:
    hidden_size = 576 # 9 * 64
    num_attention_heads = 9
    num_key_value_heads = 3
    vocab_size = 1000
    num_hidden_layers = 2
    rms_norm_eps = 1e-5
    max_position_embeddings = 1024
    intermediate_size = 1536
    
    # DeepSeek params
    q_lora_rank = 128
    kv_lora_rank = 128
    nope_head_dim = 32
    rope_head_dim = 32
    num_shared_experts = 2
    num_routed_experts = 8
    num_active_experts = 2
    expert_intermediate_size = 384

config = MockConfig()

print("Initializing Model...")
model = DeepSeekLM(config)
print("Model Initialized.")

# Test Forward Pass
input_ids = torch.randint(0, 1000, (2, 64)) # bsz=2, seq_len=64
print("Running Forward Pass...")
logits, expert_usage = model(input_ids)

print(f"Logits Shape: {logits.shape}")
print(f"Expert Usage: {expert_usage}")

assert logits.shape == (2, 64, 1000)
assert expert_usage.shape == (8,) # 8 routed experts

print("Verification Successful!")
