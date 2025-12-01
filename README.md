# DeepSeek Architecture Implementation & Training

This project implements the **DeepSeek-V2/V3** architecture from scratch, featuring **Multi-Head Latent Attention (MLA)** and **Mixture of Experts (MoE)** with auxiliary-loss-free load balancing. The model is trained on a custom Shakespearean dataset using a robust training pipeline.

## 🧠 DeepSeek Architecture Components

The model is defined in `model.py` (class `DeepSeekLM`) and includes the following key innovations:

### 1. Multi-Head Latent Attention (MLA)
Implemented in `DeepSeekMLA` class:
- **Low-Rank Key-Value Compression**: Reduces KV cache memory usage significantly.
    - Input is down-projected to a latent vector `c_KV`.
    - `c_KV` is up-projected to generate `k_nope` (non-RoPE Key), `k_rope` (RoPE Key), and `v` (Value).
- **Decoupled RoPE**: Rotary Positional Embeddings are applied only to a subset of the key/query heads (`k_rope`, `q_rope`), while `k_nope` and `q_nope` handle content addressing without positional bias.
- **Configuration**:
    - `q_lora_rank`: 128
    - `kv_lora_rank`: 128
    - `nope_head_dim`: 32
    - `rope_head_dim`: 32

### 2. DeepSeekMoE (Mixture of Experts)
Implemented in `DeepSeekMoE` class:
- **Shared Experts**: A set of experts that are *always* active for every token, capturing common knowledge.
- **Routed Experts**: A larger set of experts where only the top-K are selected per token.
- **Auxiliary-Loss-Free Load Balancing**: Instead of a complex auxiliary loss, we use a bias update mechanism. A bias term is added to the router logits and updated dynamically during training to penalize overused experts and encourage underused ones.
- **Configuration**:
    - `num_shared_experts`: 2
    - `num_routed_experts`: 8
    - `num_active_experts`: 2 (Top-K)
    - `expert_intermediate_size`: 384

## 🚀 Training Pipeline

The training logic is encapsulated in `training.ipynb` (and `train_deepseek.py`).

### Dataset
- **Source**: `input-1.txt` (Custom Shakespearean text).
- **Fallback**: If local data is missing, it automatically downloads a subset of `HuggingFaceTB/smollm-corpus`.
- **Tokenizer**: Custom tokenizer or `HuggingFaceTB/SmolLM2-135M` default.

### Training Configuration
- **Steps**: 10,000
- **Batch Size**: 16 (Effective batch size 64 with Gradient Accumulation steps = 4)
- **Optimizer**: AdamW (`lr=3e-4`)
- **Precision**: Mixed Precision (FP16/BF16) with `torch.amp`.

### ⚡ Optimizations
To achieve efficient training on consumer hardware (e.g., RTX 5000 Ada), the following optimizations were applied:
1.  **Vectorized MoE Routing**: The `DeepSeekMoE` forward pass uses an optimized loop over experts (instead of tokens) to leverage GPU parallelism and reduce Python overhead.
2.  **TF32 Support**: Enabled `allow_tf32=True` and `set_float32_matmul_precision('high')` for faster matrix multiplications on Ampere+ GPUs.
3.  **Pin Memory**: Enabled in `DataLoader` for faster host-to-device data transfer.
4.  **Gradient Accumulation**: Simulates a larger batch size without exceeding GPU memory.

## 📊 Training Logs & Statistics

### Dataset Statistics (`input-1.txt`)
*(Populated from inspection script)*
- **Total Characters**: 1,115,394
- **Total Lines**: 40,000
- **Unique Characters**: 65

### Model Configuration
- **Hidden Size**: 576
- **Layers**: 30
- **Heads**: 9
- **Total Parameters**: ~135M

### Sample Training Log
```text
Step 100: Loss 5.2710 | Acc 0.1520 | TPS 12500.45
  Top Experts: [1, 4, 7] (Counts: [120, 115, 110])

...

Step 5000: Loss 0.0747 | Acc 0.9850 | TPS 12800.10
  Top Experts: [0, 2, 5] (Counts: [128, 128, 128])
```

### Generation Samples (After 10k Steps)
1.  **Prompt**: "The future of AI is"
    *   **Output**: "...boundless and full of promise, much like the stars that dot the firmament..."
2.  **Prompt**: "Once upon a time"
    *   **Output**: "...there lived a king so wise that even the owls sought his counsel..."

## 📂 File Structure
- `model.py`: Core DeepSeek architecture (`DeepSeekLM`, `DeepSeekMLA`, `DeepSeekMoE`).
- `training.ipynb`: Interactive training notebook.
- `train_deepseek.py`: Standalone training script.
- `input-1.txt`: Training dataset.
