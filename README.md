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

- **Total Characters**: 1,115,394
- **Total Lines**: 40,000
- **Unique Characters**: 65

### Model Configuration
- **Hidden Size**: 576
- **Layers**: 30
- **Heads**: 9
- **Total Parameters**: ~135M

###Training Log
```text
DeepSeekLM(
  (embed_tokens): Embedding(49152, 576)
  (layers): ModuleList(
    (0-29): 30 x Block(
      (self_attn): DeepSeekMLA(
        (kv_down_proj): Linear(in_features=576, out_features=128, bias=False)
        (kv_norm): RMSNorm()
        (w_uk): Linear(in_features=128, out_features=288, bias=False)
        (w_ur): Linear(in_features=128, out_features=288, bias=False)
        (w_uv): Linear(in_features=128, out_features=576, bias=False)
        (q_down_proj): Linear(in_features=576, out_features=128, bias=False)
        (q_norm): RMSNorm()
        (w_uq): Linear(in_features=128, out_features=288, bias=False)
        (w_qr): Linear(in_features=128, out_features=288, bias=False)
        (o_proj): Linear(in_features=576, out_features=576, bias=False)
      )
      (mlp): DeepSeekMoE(
        (shared_experts): ModuleList(
          (0): DeepSeekExpertLayer(
            (gate_proj): Linear(in_features=576, out_features=384, bias=False)
            (up_proj): Linear(in_features=576, out_features=384, bias=False)
            (down_proj): Linear(in_features=384, out_features=576, bias=False)
            (act_fn): SiLU()
          )
        )
...
  )
  (norm): RMSNorm()
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
Skipping torch.compile on Windows to avoid potential compatibility issues.
Starting training...
Step 100: Loss 4.6934 | Acc 0.2505 | TPS 4007.58
  Top Experts: [3, 1, 6] (Counts: [36809.0, 33519.0, 32624.0])
Step 200: Loss 4.0469 | Acc 0.3025 | TPS 4174.99
  Top Experts: [3, 1, 6] (Counts: [35844.0, 33733.0, 33348.0])
Step 300: Loss 3.7518 | Acc 0.3353 | TPS 4202.27
  Top Experts: [1, 3, 6] (Counts: [33874.0, 33787.0, 33483.0])
Step 400: Loss 3.4933 | Acc 0.3642 | TPS 4165.01
  Top Experts: [3, 1, 0] (Counts: [34359.0, 33296.0, 32160.0])
Step 500: Loss 3.1531 | Acc 0.4066 | TPS 4132.45
  Top Experts: [3, 1, 6] (Counts: [35348.0, 33343.0, 32253.0])
Step 600: Loss 2.2389 | Acc 0.5588 | TPS 4147.06
  Top Experts: [3, 6, 1] (Counts: [34546.0, 32690.0, 32171.0])
Step 700: Loss 1.9544 | Acc 0.6103 | TPS 4124.61
  Top Experts: [1, 3, 6] (Counts: [34124.0, 33445.0, 31511.0])
Step 800: Loss 1.4430 | Acc 0.7078 | TPS 4117.61
  Top Experts: [1, 6, 3] (Counts: [33659.0, 33184.0, 32558.0])
Step 900: Loss 0.9985 | Acc 0.7941 | TPS 4135.36
  Top Experts: [1, 3, 0] (Counts: [34557.0, 34180.0, 32137.0])
Step 1000: Loss 0.5874 | Acc 0.8880 | TPS 4159.42
  Top Experts: [3, 1, 0] (Counts: [33242.0, 33218.0, 32257.0])

--- Step 1000 Generation ---
Generated: The meaning of life is now dead?

NORTHUMBERLAND:
No, my lord, nothing but this shame,
At last I send me, though to seek loss,
Shall be the face, and look on Henry's life.
-----------------------------

Step 1100: Loss 0.1446 | Acc 0.9838 | TPS 4081.54
  Top Experts: [3, 1, 0] (Counts: [33201.0, 32888.0, 32215.0])
Step 1200: Loss 0.0846 | Acc 0.9863 | TPS 4097.88
  Top Experts: [3, 6, 1] (Counts: [32490.0, 32258.0, 31816.0])
Step 1300: Loss 0.0468 | Acc 0.9922 | TPS 4107.09
  Top Experts: [3, 1, 0] (Counts: [33978.0, 32864.0, 32625.0])
Step 1400: Loss 0.0262 | Acc 0.9958 | TPS 4120.77
  Top Experts: [6, 0, 3] (Counts: [32732.0, 32582.0, 31708.0])
Step 1500: Loss 0.0187 | Acc 0.9975 | TPS 4128.73
  Top Experts: [0, 1, 3] (Counts: [34485.0, 31924.0, 31604.0])
Step 1600: Loss 0.0168 | Acc 0.9966 | TPS 4142.95
  Top Experts: [3, 4, 1] (Counts: [31829.0, 31773.0, 31739.0])
Step 1700: Loss 0.0219 | Acc 0.9956 | TPS 4170.94
  Top Experts: [3, 1, 0] (Counts: [33449.0, 32459.0, 31572.0])
Step 1800: Loss 0.0122 | Acc 0.9980 | TPS 4196.24
  Top Experts: [3, 0, 1] (Counts: [33347.0, 32167.0, 31416.0])
Step 1900: Loss 0.0132 | Acc 0.9973 | TPS 4204.00
  Top Experts: [1, 3, 6] (Counts: [33147.0, 32207.0, 31234.0])
Step 2000: Loss 0.0177 | Acc 0.9961 | TPS 4208.44
  Top Experts: [1, 3, 4] (Counts: [32510.0, 31876.0, 31239.0])

--- Step 2000 Generation ---
Generated: The meaning of life isle,
That by the justice of true judgment-ple'd out,
When heinous article, justice of what he lies
And what he bites, that must be set,
A sleeping England and lief-broke repeals,
-----------------------------

Step 2100: Loss 0.0169 | Acc 0.9971 | TPS 4171.85
  Top Experts: [1, 0, 4] (Counts: [8186.0, 7999.0, 7872.0])
Step 2200: Loss 0.0190 | Acc 0.9963 | TPS 4173.85
  Top Experts: [6, 3, 5] (Counts: [32350.0, 31431.0, 30997.0])
Step 2300: Loss 0.0452 | Acc 0.9941 | TPS 4176.96
  Top Experts: [1, 0, 3] (Counts: [33864.0, 32333.0, 31734.0])
Step 2400: Loss 0.3885 | Acc 0.9184 | TPS 4174.06
  Top Experts: [3, 4, 6] (Counts: [32086.0, 31306.0, 31299.0])
Step 2500: Loss 0.5854 | Acc 0.8588 | TPS 4179.81
  Top Experts: [3, 1, 6] (Counts: [32751.0, 32180.0, 31557.0])
Step 2600: Loss 0.1732 | Acc 0.9618 | TPS 4185.42
  Top Experts: [1, 4, 3] (Counts: [32360.0, 32266.0, 31473.0])
Step 2700: Loss 0.0365 | Acc 0.9939 | TPS 4180.68
  Top Experts: [6, 3, 2] (Counts: [32787.0, 32068.0, 31322.0])
Step 2800: Loss 0.0217 | Acc 0.9963 | TPS 4185.05
  Top Experts: [3, 4, 6] (Counts: [32959.0, 32066.0, 31313.0])
Step 2900: Loss 0.0198 | Acc 0.9958 | TPS 4192.67
  Top Experts: [0, 1, 4] (Counts: [31727.0, 31579.0, 31517.0])
Step 3000: Loss 0.0146 | Acc 0.9963 | TPS 4191.38
  Top Experts: [3, 6, 0] (Counts: [31895.0, 31704.0, 31341.0])

--- Step 3000 Generation ---
Generated: The meaning of life isle'd,
In London by the tempest to this land of heaven:
Be not the holy sacrament, is the motive
Of this small inferior life.

LEONTES:
Will't please your silence,--

LE
-----------------------------

Step 3100: Loss 0.0102 | Acc 0.9980 | TPS 4130.37
  Top Experts: [4, 6, 1] (Counts: [32418.0, 32246.0, 31353.0])
Step 3200: Loss 0.0082 | Acc 0.9971 | TPS 4138.08
  Top Experts: [3, 6, 4] (Counts: [31501.0, 31420.0, 31036.0])
Step 3300: Loss 0.0107 | Acc 0.9968 | TPS 4145.70
  Top Experts: [3, 1, 2] (Counts: [31818.0, 31298.0, 31284.0])
Step 3400: Loss 0.0116 | Acc 0.9968 | TPS 4149.21
  Top Experts: [6, 3, 4] (Counts: [32225.0, 31946.0, 31304.0])
Step 3500: Loss 0.0075 | Acc 0.9980 | TPS 4152.85
  Top Experts: [3, 4, 0] (Counts: [32273.0, 31274.0, 31180.0])
Step 3600: Loss 0.0087 | Acc 0.9980 | TPS 4153.04
  Top Experts: [4, 6, 1] (Counts: [32157.0, 31398.0, 31264.0])
Step 3700: Loss 0.0093 | Acc 0.9973 | TPS 4152.48
  Top Experts: [6, 2, 1] (Counts: [32011.0, 32010.0, 31756.0])
Step 3800: Loss 0.0081 | Acc 0.9978 | TPS 4154.09
  Top Experts: [4, 1, 2] (Counts: [31715.0, 31666.0, 31351.0])
Step 3900: Loss 0.0162 | Acc 0.9958 | TPS 4158.75
  Top Experts: [1, 6, 0] (Counts: [31783.0, 31383.0, 31335.0])
Step 4000: Loss 0.0152 | Acc 0.9966 | TPS 4166.97
  Top Experts: [4, 1, 0] (Counts: [31832.0, 31728.0, 31423.0])

--- Step 4000 Generation ---
Generated: The meaning of life isle,
In gross blood to this shame; till so be thou,
Thou art thou but the life of the life,
And this poor mortal company, this wretch,
Is it not wrong'd with the point of grief

-----------------------------

Step 4100: Loss 0.0103 | Acc 0.9971 | TPS 4146.26
  Top Experts: [6, 3, 1] (Counts: [31363.0, 31254.0, 30904.0])
Step 4200: Loss 0.0107 | Acc 0.9980 | TPS 4146.69
  Top Experts: [1, 2, 3] (Counts: [8211.0, 8098.0, 8008.0])
Step 4300: Loss 0.0102 | Acc 0.9968 | TPS 4153.38
  Top Experts: [3, 6, 0] (Counts: [32714.0, 31615.0, 31491.0])
Step 4400: Loss 0.0095 | Acc 0.9968 | TPS 4157.83
  Top Experts: [6, 1, 2] (Counts: [32323.0, 31541.0, 31308.0])
Step 4500: Loss 0.0115 | Acc 0.9968 | TPS 4168.02
  Top Experts: [3, 4, 0] (Counts: [31788.0, 30926.0, 30906.0])
Step 4600: Loss 0.0109 | Acc 0.9971 | TPS 4180.91
  Top Experts: [1, 2, 3] (Counts: [31629.0, 31566.0, 31278.0])
Step 4700: Loss 0.0146 | Acc 0.9961 | TPS 4194.30
  Top Experts: [3, 0, 1] (Counts: [32227.0, 32100.0, 31092.0])
Step 4800: Loss 0.0080 | Acc 0.9973 | TPS 4198.21
  Top Experts: [6, 1, 2] (Counts: [31504.0, 31225.0, 31208.0])
Step 4900: Loss 0.0068 | Acc 0.9980 | TPS 4205.77
  Top Experts: [7, 6, 5] (Counts: [31519.0, 31469.0, 31307.0])
Step 5000: Loss 0.0136 | Acc 0.9966 | TPS 4215.92
  Top Experts: [6, 3, 5] (Counts: [31687.0, 31548.0, 31389.0])

--- Step 5000 Generation ---
Generated: The meaning of life isle,
Of the sky-dro thee and haste
Of this vexation, when 'tis true that very hour
Than pity to it! no more: so look thee,
The precedent doth harbour such a nettle-
-----------------------------

Step 5100: Loss 0.0108 | Acc 0.9971 | TPS 4209.08
  Top Experts: [7, 3, 4] (Counts: [31259.0, 31062.0, 31060.0])
Step 5200: Loss 0.0145 | Acc 0.9958 | TPS 4214.50
  Top Experts: [3, 7, 1] (Counts: [32226.0, 31464.0, 31053.0])
Step 5300: Loss 0.0097 | Acc 0.9973 | TPS 4219.48
  Top Experts: [3, 1, 5] (Counts: [31887.0, 31766.0, 31334.0])
Step 5400: Loss 0.0080 | Acc 0.9975 | TPS 4226.18
  Top Experts: [6, 0, 7] (Counts: [32002.0, 31336.0, 31304.0])
Step 5500: Loss 0.0105 | Acc 0.9975 | TPS 4231.86
  Top Experts: [1, 3, 5] (Counts: [32079.0, 31140.0, 30991.0])
Step 5600: Loss 1.3267 | Acc 0.6716 | TPS 4233.19
  Top Experts: [0, 6, 3] (Counts: [32370.0, 31243.0, 31161.0])
Step 5700: Loss 0.6528 | Acc 0.8373 | TPS 4242.16
  Top Experts: [4, 0, 5] (Counts: [31643.0, 31510.0, 31047.0])
Step 5800: Loss 0.0447 | Acc 0.9897 | TPS 4245.72
  Top Experts: [4, 0, 6] (Counts: [32258.0, 31181.0, 31127.0])
Step 5900: Loss 0.0314 | Acc 0.9931 | TPS 4250.01
  Top Experts: [1, 0, 6] (Counts: [32239.0, 32009.0, 31601.0])
Step 6000: Loss 0.0184 | Acc 0.9953 | TPS 4254.31
  Top Experts: [4, 0, 1] (Counts: [31888.0, 31540.0, 31168.0])

--- Step 6000 Generation ---
Generated: The meaning of life isle and mock you.

MENENIUS:
If you be patient, I'll try how I'll try how
you means, power still without givingiest.

BRUTUS:
Not in the party of your
-----------------------------

Step 6100: Loss 0.0144 | Acc 0.9961 | TPS 4246.98
  Top Experts: [0, 4, 1] (Counts: [33216.0, 31075.0, 30729.0])
Step 6200: Loss 0.0067 | Acc 0.9985 | TPS 4250.57
  Top Experts: [4, 7, 6] (Counts: [32680.0, 31546.0, 30947.0])
Step 6300: Loss 0.0242 | Acc 0.9951 | TPS 4251.02
  Top Experts: [5, 4, 0] (Counts: [8225.0, 8079.0, 8044.0])
Step 6400: Loss 0.0106 | Acc 0.9971 | TPS 4252.75
  Top Experts: [6, 1, 5] (Counts: [31966.0, 31162.0, 31098.0])
Step 6500: Loss 0.0118 | Acc 0.9968 | TPS 4248.87
  Top Experts: [4, 6, 5] (Counts: [32524.0, 31618.0, 31207.0])
Step 6600: Loss 0.0092 | Acc 0.9975 | TPS 4248.54
  Top Experts: [0, 4, 3] (Counts: [31807.0, 31462.0, 31192.0])
Step 6700: Loss 0.0095 | Acc 0.9973 | TPS 4248.98
  Top Experts: [2, 0, 6] (Counts: [32218.0, 31820.0, 31564.0])
Step 6800: Loss 0.0144 | Acc 0.9966 | TPS 4250.35
  Top Experts: [4, 6, 0] (Counts: [32110.0, 32046.0, 31726.0])
Step 6900: Loss 0.0103 | Acc 0.9971 | TPS 4250.97
  Top Experts: [3, 2, 7] (Counts: [31207.0, 31185.0, 31002.0])
Step 7000: Loss 0.0095 | Acc 0.9971 | TPS 4253.28
  Top Experts: [0, 6, 2] (Counts: [32347.0, 31590.0, 31396.0])

--- Step 7000 Generation ---
Generated: The meaning of life isle and fruit fought,
And mark me we from this virtuous tears,
And you shall faint from wedlock judgment-like betroddenited.
This swells the siege of this fair prayer
And tell him that hath lost for bad
-----------------------------

Step 7100: Loss 0.0118 | Acc 0.9963 | TPS 4240.34
  Top Experts: [4, 3, 7] (Counts: [32111.0, 31613.0, 31485.0])
Step 7200: Loss 0.0104 | Acc 0.9966 | TPS 4241.01
  Top Experts: [6, 0, 7] (Counts: [32273.0, 31499.0, 31144.0])
Step 7300: Loss 0.0150 | Acc 0.9956 | TPS 4243.53
  Top Experts: [2, 0, 3] (Counts: [32130.0, 31461.0, 30738.0])
Step 7400: Loss 0.0057 | Acc 0.9978 | TPS 4245.77
  Top Experts: [4, 5, 6] (Counts: [31879.0, 31120.0, 31001.0])
Step 7500: Loss 0.0048 | Acc 0.9985 | TPS 4247.04
  Top Experts: [0, 1, 3] (Counts: [32428.0, 31633.0, 30684.0])
Step 7600: Loss 0.0107 | Acc 0.9973 | TPS 4250.12
  Top Experts: [4, 0, 2] (Counts: [31535.0, 31476.0, 31469.0])
Step 7700: Loss 0.0130 | Acc 0.9966 | TPS 4250.85
  Top Experts: [0, 6, 2] (Counts: [31968.0, 31786.0, 31377.0])
Step 7800: Loss 0.0162 | Acc 0.9958 | TPS 4253.67
  Top Experts: [3, 6, 4] (Counts: [32232.0, 30888.0, 30887.0])
Step 7900: Loss 0.0037 | Acc 0.9988 | TPS 4251.89
  Top Experts: [7, 5, 0] (Counts: [31259.0, 30906.0, 30851.0])
Step 8000: Loss 0.0100 | Acc 0.9968 | TPS 4251.39
  Top Experts: [6, 0, 4] (Counts: [32235.0, 31853.0, 31470.0])

--- Step 8000 Generation ---
Generated: The meaning of life isle and record,
But that we oweed much strength to confutes.

BUSHENRY PERCY:
My lord, I am going to the goal;
And long'd it is the issue:
This is the
-----------------------------

Step 8100: Loss 0.0092 | Acc 0.9968 | TPS 4240.68
  Top Experts: [0, 3, 7] (Counts: [32540.0, 31478.0, 31353.0])
Step 8200: Loss 0.0074 | Acc 0.9978 | TPS 4241.94
  Top Experts: [1, 6, 4] (Counts: [32136.0, 31805.0, 31606.0])
Step 8300: Loss 0.0118 | Acc 0.9963 | TPS 4089.25
  Top Experts: [1, 6, 2] (Counts: [32083.0, 31726.0, 31645.0])
Step 8400: Loss 0.0172 | Acc 0.9941 | TPS 3907.54
  Top Experts: [4, 2, 6] (Counts: [8092.0, 8054.0, 7937.0])
Step 8500: Loss 0.0102 | Acc 0.9968 | TPS 3765.16
  Top Experts: [4, 6, 2] (Counts: [31729.0, 31629.0, 31591.0])
Step 8600: Loss 0.0094 | Acc 0.9968 | TPS 3774.17
  Top Experts: [4, 1, 3] (Counts: [32896.0, 31350.0, 31273.0])
Step 8700: Loss 0.0123 | Acc 0.9971 | TPS 3781.50
  Top Experts: [5, 3, 7] (Counts: [31082.0, 31071.0, 31018.0])
Step 8800: Loss 0.0357 | Acc 0.9912 | TPS 3789.98
  Top Experts: [3, 6, 0] (Counts: [32023.0, 31713.0, 31389.0])
Step 8900: Loss 0.9243 | Acc 0.7471 | TPS 3795.72
  Top Experts: [0, 6, 1] (Counts: [32457.0, 31687.0, 31443.0])
Step 9000: Loss 0.1087 | Acc 0.9725 | TPS 3800.51
  Top Experts: [0, 6, 2] (Counts: [33510.0, 31714.0, 31015.0])

--- Step 9000 Generation ---
Generated: The meaning of life isle:
But, for his own affections' means,
Which grieves to be suppliken!

Second Murderer:
'Zounds, he does sit in the reward,
And not his house looker so
-----------------------------

Step 9100: Loss 0.0328 | Acc 0.9929 | TPS 3796.31
  Top Experts: [0, 2, 7] (Counts: [32007.0, 31803.0, 31204.0])
Step 9200: Loss 0.0136 | Acc 0.9958 | TPS 3801.69
  Top Experts: [6, 0, 2] (Counts: [31495.0, 31460.0, 31325.0])
Step 9300: Loss 0.0168 | Acc 0.9953 | TPS 3807.41
  Top Experts: [5, 6, 0] (Counts: [32709.0, 31343.0, 31130.0])
Step 9400: Loss 0.0136 | Acc 0.9968 | TPS 3812.82
  Top Experts: [6, 1, 0] (Counts: [32269.0, 31817.0, 31136.0])
Step 9500: Loss 0.0120 | Acc 0.9961 | TPS 3816.78
  Top Experts: [2, 0, 1] (Counts: [31547.0, 31362.0, 30832.0])
Step 9600: Loss 0.0069 | Acc 0.9983 | TPS 3821.84
  Top Experts: [2, 0, 6] (Counts: [32781.0, 31838.0, 31271.0])
Step 9700: Loss 0.0129 | Acc 0.9963 | TPS 3826.12
  Top Experts: [1, 0, 3] (Counts: [31768.0, 31009.0, 30989.0])
Step 9800: Loss 0.0103 | Acc 0.9968 | TPS 3832.14
  Top Experts: [5, 0, 6] (Counts: [31817.0, 31542.0, 31316.0])
Step 9900: Loss 0.0133 | Acc 0.9963 | TPS 3836.35
  Top Experts: [6, 2, 1] (Counts: [31872.0, 31545.0, 31369.0])
Step 10000: Loss 0.0083 | Acc 0.9978 | TPS 3839.70
  Top Experts: [5, 4, 6] (Counts: [32753.0, 31491.0, 31143.0])

--- Step 10000 Generation ---
Generated: The meaning of life isle of thee,
That thou wert possess'd Richard'st degree,
Is far off so happy by his pilgrimage;
And yet, madam, he is held forsworn.

KING RICHARD II:

-----------------------------

Checkpoint saved to checkpoint_10000.pt
```

### Generation Output (After 10k Steps)


        
        Generation 1:

        Generated: The future of AI is hot;
        Who shall be the vantage of that,
        Who wooer the vantage of the world,
        Make me, for a day to fight,
        When he did thy woe; but in those things
        After the blushing blood in the world,
        Call for a, not one infect another,
        From your holy kingly on the world,
        Be ready to't, or if thou wilt the world,
        Is more but green where I thy dost confess,

        
        Generation 2:
        Generated: Once upon a time;
        And what we are, that they shall be tears.
        
        HENRY BOLINGBROKE:
        Thou counterfeit'st a woe to make me ask.
        
        KING RICHARD II:
        Northumberland, thou hast resisted; the breath of care?
        
        DUKE OF AUMERLE:
        Fitzwater, thou art all things to be satisfied.
        
        KING RICHARD II:
        Thou art too careless
        
        Generation 3:
        Generated: In a galaxy far away.
        
        THOMAS MOWBRAY:
        Then thus I turn me from my country's light,
        To dwell in solemn shades of endless night.
        
        KING RICHARD II:
        Return again, and take an oath with thee.
        Lay on our royal sword your banish'd hands;
        Swear by the duty that you owe to God--
        Our part therein we banish with yourselves--
        To keep the oath that we administer:
        You never shall
        
        Generation 4:
        Generated: The secret to happiness is vain:
        Thou art most likely; and not the noise of night
        And all the swift passage polts,
        Concern me, three shepherds looking on the night,
        Like to the dead bodies of the devil's top,
        And time the fadvised her friends':
        Make motion through the sun under fiends roar'd.
        This is the duke, that's gone roundly
        To the dead bodies of the envious siege
        Of watery Neptune, and not our
        
        Generation 5:

        Generated: Python is a programming language that I know,
        Still such a goodly have been more than in my fault,
        And, if you would not this night, I'll be gone:
        And since this time, alack, for your brother,
        And, in this time, is no farther offend,
        And in the pate of those sciences,
        That it should die before your brother's hate.
        
        GRUMIO:
        O, madam, keep mine eye the time of wooing
        -------------------------------------

## 📂 File Structure
- `model.py`: Core DeepSeek architecture (`DeepSeekLM`, `DeepSeekMLA`, `DeepSeekMoE`).
- `training.ipynb`: Interactive training notebook.
- `train_deepseek.py`: Standalone training script.
- `input-1.txt`: Training dataset.



