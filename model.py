
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = (x.pow(2).mean(-1, keepdim=True))
        x = x * torch.rsqrt(mean_square + self.eps)
        return self.weight * x

def rotate_half(x):
    # Rotates half the hidden dims of the input.
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [bsz, heads, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim] -> unsqueeze to [1, 1, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DeepSeekExpertLayer(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        # Allow overriding intermediate size for experts
        inter_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter_size, bias=False)
        self.down_proj = nn.Linear(inter_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMLA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", 128)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 128)
        self.nope_head_dim = getattr(config, "nope_head_dim", 32)
        self.rope_head_dim = getattr(config, "rope_head_dim", 32)
        self.head_dim = self.nope_head_dim + self.rope_head_dim
        
        # Compressed KV
        self.kv_down_proj = nn.Linear(self.hidden_size, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.w_uk = nn.Linear(self.kv_lora_rank, self.num_heads * self.nope_head_dim, bias=False) # k_nope
        self.w_ur = nn.Linear(self.kv_lora_rank, self.num_heads * self.rope_head_dim, bias=False) # k_rope
        self.w_uv = nn.Linear(self.kv_lora_rank, self.num_heads * self.head_dim, bias=False)      # v

        # Compressed Q
        self.q_down_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.w_uq = nn.Linear(self.q_lora_rank, self.num_heads * self.nope_head_dim, bias=False) # q_nope
        self.w_qr = nn.Linear(self.q_lora_rank, self.num_heads * self.rope_head_dim, bias=False) # q_rope
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cos, sin, mask=None):
        bsz, seq_len, _ = x.shape
        
        # KV Compression
        c_kv = self.kv_norm(self.kv_down_proj(x))
        k_nope = self.w_uk(c_kv).view(bsz, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        k_rope = self.w_ur(c_kv).view(bsz, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Q Compression
        c_q = self.q_norm(self.q_down_proj(x))
        q_nope = self.w_uq(c_q).view(bsz, seq_len, self.num_heads, self.nope_head_dim).transpose(1, 2)
        q_rope = self.w_qr(c_q).view(bsz, seq_len, self.num_heads, self.rope_head_dim).transpose(1, 2)
        
        # Apply RoPE
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        
        # Concatenate
        q = torch.cat([q_nope, q_rope], dim=-1) # [bsz, num_heads, seq_len, head_dim]
        k = torch.cat([k_nope, k_rope], dim=-1) # [bsz, num_heads, seq_len, head_dim]
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = attn_weights + mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        return self.o_proj(output)

class DeepSeekMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_shared_experts = getattr(config, "num_shared_experts", 2)
        self.num_routed_experts = getattr(config, "num_routed_experts", 8)
        self.num_active_experts = getattr(config, "num_active_experts", 2)
        self.intermediate_size = getattr(config, "expert_intermediate_size", 384)
        
        # Shared Experts
        self.shared_experts = nn.ModuleList([
            DeepSeekExpertLayer(config, intermediate_size=self.intermediate_size) 
            for _ in range(self.num_shared_experts)
        ])
        
        # Routed Experts
        self.routed_experts = nn.ModuleList([
            DeepSeekExpertLayer(config, intermediate_size=self.intermediate_size)
            for _ in range(self.num_routed_experts)
        ])
        
        self.router = nn.Linear(self.hidden_size, self.num_routed_experts, bias=False)
        
        # Load Balancing Bias (Auxiliary-loss-free)
        self.register_buffer("router_bias", torch.zeros(self.num_routed_experts))
        self.bias_update_rate = 0.001 

    def forward(self, x):
        bsz, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Shared Experts
        shared_out = sum([e(x) for e in self.shared_experts])
        
        # Router
        router_logits = self.router(x) 
        router_logits_flat = router_logits.view(-1, self.num_routed_experts)
        
        biased_logits = router_logits_flat + self.router_bias
        
        # Top-K
        topk_weights, topk_indices = torch.topk(torch.sigmoid(biased_logits), self.num_active_experts, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True) 
        
        # Route
        final_routed_out = torch.zeros_like(x_flat)
        expert_usage = torch.zeros(self.num_routed_experts, device=x.device)
        
        # Optimized Loop: Iterate over experts instead of K
        # This is generally faster when num_routed_experts is small (e.g. 8)
        for expert_idx in range(self.num_routed_experts):
            # Find which tokens selected this expert (in any of the k positions)
            # topk_indices: [tokens, k]
            # We create a mask for tokens that selected this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1) # [tokens]
            
            if expert_mask.any():
                expert_usage[expert_idx] = expert_mask.sum()
                
                # Select tokens
                selected_x = x_flat[expert_mask]
                
                # Compute expert output
                expert_out = self.routed_experts[expert_idx](selected_x) # [selected_tokens, hidden]
                
                # We need to add this back to final_routed_out, weighted by the correct weight.
                # A token might select this expert at position k=0 or k=1...
                # We need to find the weight for this expert for each selected token.
                
                # Get weights for this expert for the selected tokens
                # topk_indices[expert_mask] -> [selected_tokens, k]
                # topk_weights[expert_mask] -> [selected_tokens, k]
                
                current_indices = topk_indices[expert_mask]
                current_weights = topk_weights[expert_mask]
                
                # Create a mask for where the expert is in top-k
                # (current_indices == expert_idx) -> [selected_tokens, k]
                # We sum weights across k (should be only one match usually, but safe to sum)
                weight_mask = (current_indices == expert_idx).float()
                scaling = (current_weights * weight_mask).sum(dim=-1, keepdim=True) # [selected_tokens, 1]
                
                final_routed_out[expert_mask] += expert_out * scaling
                    
        # Update Bias (Training only)
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(topk_indices, num_classes=self.num_routed_experts).float()
                usage = one_hot.sum(dim=(0, 1))
                usage_prob = usage / (usage.sum() + 1e-6)
                
                target_prob = 1.0 / self.num_routed_experts
                error = usage_prob - target_prob
                self.router_bias -= self.bias_update_rate * torch.sign(error)

        return (shared_out + final_routed_out.view(bsz, seq_len, -1)), expert_usage

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = DeepSeekMLA(config)
        self.mlp = DeepSeekMoE(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, mask=None):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        moe_out, expert_usage = self.mlp(self.post_attention_layernorm(h))
        out = h + moe_out
        return out, expert_usage

class DeepSeekLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # RoPE setup
        self.rope_head_dim = getattr(config, "rope_head_dim", 32)
        self.head_dim = self.rope_head_dim
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.max_pos = config.max_position_embeddings * 2
        self._set_cos_sin_cache(self.max_pos)

    def _set_cos_sin_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        x = self.embed_tokens(input_ids)

        if self.cos_cached.device != x.device or self.cos_cached.shape[0] < seq_len:
            self.inv_freq = self.inv_freq.to(x.device)
            self._set_cos_sin_cache(max(seq_len, 2048))

        cos = self.cos_cached[:seq_len].to(dtype=x.dtype)
        sin = self.sin_cached[:seq_len].to(dtype=x.dtype)

        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            mask = torch.triu(mask, diagonal=1)

        total_expert_usage = None
        
        for layer in self.layers:
            x, expert_usage = layer(x, cos, sin, mask)
            if total_expert_usage is None:
                total_expert_usage = expert_usage
            else:
                total_expert_usage += expert_usage

        x = self.norm(x)
        logits = self.lm_head(x)
        
        if self.training:
            return logits, total_expert_usage
        else:
            return logits
