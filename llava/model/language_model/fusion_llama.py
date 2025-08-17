"""
Fusion-enhanced LLaMA model implementation for vision-language tasks
"""

import inspect
import types
import math
from typing import List, Optional, Tuple, Union, Callable, TypedDict, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
# from transformers.models.llama.modeling_llama import flash_attn_func, flash_attn_varlen_func, pad_input
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)

# from transformers.processing_utils import Unpack
from transformers.utils import (
    logging,
)
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv 
from ..fusion_arch import FusionMetaModel, FusionMetaForCausalLM
from ..fusion_layer import CustomLoRAVisionFusionLayer, StableFusionLayer
from llava.constants import (
    # IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_FUSION_ALPHA,
    DEFAULT_FUSION_DROPOUT,
    DEFAULT_FUSION_TARGETS,
    DEFAULT_FUSION_VISION_DIM,
    DEFAULT_FUSION_HIDDEN_DIM,
    DEFAULT_FUSION_LAYER_IDS_TO_INJECT
)

logger = logging.get_logger(__name__)

##### Rewrite the decoder layer to include fusion
# import math
# from typing import List, Optional, Tuple, Union

# import torch
# import torch.nn.functional as F
# import torch.utils.checkpoint
# from torch import nn
# from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers.activations import ACT2FN
# from transformers.cache_utils import Cache, DynamicCache, StaticCache
# from transformers.modeling_attn_mask_utils import AttentionMaskConverter
# from transformers.modeling_outputs import (
#     BaseModelOutputWithPast,
#     CausalLMOutputWithPast,
#     QuestionAnsweringModelOutput,
#     SequenceClassifierOutputWithPast,
# )
# from transformers.modeling_utils import PreTrainedModel
# from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
# from transformers.utils import (
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     is_flash_attn_2_available,
#     is_flash_attn_greater_or_equal_2_10,
#     logging,
#     replace_return_docstrings,
# )
# from transformers.models.llama.configuration_llama import LlamaConfig


# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


# logger = logging.get_logger(__name__)

# _CONFIG_FOR_DOC = "LlamaConfig"


# def _get_unpad_data(attention_mask):
#     seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
#     indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
#     max_seqlen_in_batch = seqlens_in_batch.max().item()
#     cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
#     return (
#         indices,
#         cu_seqlens,
#         max_seqlen_in_batch,
#     )


# class LlamaRMSNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-6):
#         """
#         LlamaRMSNorm is equivalent to T5LayerNorm
#         """
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to(torch.float32)
#         variance = hidden_states.pow(2).mean(-1, keepdim=True)
#         hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)


# ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


# class LlamaRotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
#         super().__init__()
#         self.scaling_factor = scaling_factor
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)
#         # For BC we register cos and sin cached
#         self.max_seq_len_cached = max_position_embeddings

#     @torch.no_grad()
#     def forward(self, x, position_ids):
#         # x: [bs, num_attention_heads, seq_len, head_size]
#         inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
#         position_ids_expanded = position_ids[:, None, :].float()
#         # Force float32 since bfloat16 loses precision on long contexts
#         # See https://github.com/huggingface/transformers/pull/29285
#         device_type = x.device.type
#         device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
#         with torch.autocast(device_type=device_type, enabled=False):
#             freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
#             emb = torch.cat((freqs, freqs), dim=-1)
#             cos = emb.cos()
#             sin = emb.sin()
#         return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
#     """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

#     def forward(self, x, position_ids):
#         # difference to the original RoPE: a scaling factor is aplied to the position ids
#         position_ids = position_ids.float() / self.scaling_factor
#         cos, sin = super().forward(x, position_ids)
#         return cos, sin


# class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
#     """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

#     def forward(self, x, position_ids):
#         # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
#         seq_len = torch.max(position_ids) + 1
#         if seq_len > self.max_position_embeddings:
#             base = self.base * (
#                 (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
#             ) ** (self.dim / (self.dim - 2))
#             inv_freq = 1.0 / (
#                 base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
#             )
#             self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

#         cos, sin = super().forward(x, position_ids)
#         return cos, sin


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


# class LlamaMLP(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
#         self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
#         self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(self, x):
#         if self.config.pretraining_tp > 1:
#             slice = self.intermediate_size // self.config.pretraining_tp
#             gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
#             up_proj_slices = self.up_proj.weight.split(slice, dim=0)
#             down_proj_slices = self.down_proj.weight.split(slice, dim=1)

#             gate_proj = torch.cat(
#                 [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
#             )
#             up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

#             intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
#             down_proj = [
#                 F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
#             ]
#             down_proj = sum(down_proj)
#         else:
#             down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

#         return down_proj


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# class LlamaAttention(nn.Module):
#     """Multi-headed attention from 'Attention Is All You Need' paper"""

#     def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
#         super().__init__()
#         self.config = config
#         self.layer_idx = layer_idx
#         if layer_idx is None:
#             logger.warning_once(
#                 f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
#                 "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
#                 "when creating this class."
#             )

#         self.attention_dropout = config.attention_dropout
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.rope_theta = config.rope_theta
#         self.is_causal = True

#         if (self.head_dim * self.num_heads) != self.hidden_size:
#             raise ValueError(
#                 f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
#                 f" and `num_heads`: {self.num_heads})."
#             )

#         self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
#         self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
#         self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
#         self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
#         self._init_rope()

#     def _init_rope(self):
#         if self.config.rope_scaling is None:
#             self.rotary_emb = LlamaRotaryEmbedding(
#                 self.head_dim,
#                 max_position_embeddings=self.max_position_embeddings,
#                 base=self.rope_theta,
#             )
#         else:
#             scaling_type = self.config.rope_scaling["type"]
#             scaling_factor = self.config.rope_scaling["factor"]
#             if scaling_type == "linear":
#                 self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
#                     self.head_dim,
#                     max_position_embeddings=self.max_position_embeddings,
#                     scaling_factor=scaling_factor,
#                     base=self.rope_theta,
#                 )
#             elif scaling_type == "dynamic":
#                 self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
#                     self.head_dim,
#                     max_position_embeddings=self.max_position_embeddings,
#                     scaling_factor=scaling_factor,
#                     base=self.rope_theta,
#                 )
#             else:
#                 raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_value: Optional[Cache] = None,
#         output_attentions: bool = False,
#         use_cache: bool = False,
#         cache_position: Optional[torch.LongTensor] = None,
#         **kwargs,
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
#         bsz, q_len, _ = hidden_states.size()

#         if self.config.pretraining_tp > 1:
#             key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
#             query_slices = self.q_proj.weight.split(
#                 (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
#             )
#             key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
#             value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

#             query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
#             query_states = torch.cat(query_states, dim=-1)

#             key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
#             key_states = torch.cat(key_states, dim=-1)

#             value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
#             value_states = torch.cat(value_states, dim=-1)

#         else:
#             query_states = self.q_proj(hidden_states)
#             key_states = self.k_proj(hidden_states)
#             value_states = self.v_proj(hidden_states)

#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

#         cos, sin = self.rotary_emb(value_states, position_ids)
#         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

#         if past_key_value is not None:
#             # sin and cos are specific to RoPE models; cache_position needed for the static cache
#             cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
#             key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)

#         attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

#         if attention_mask is not None:  # no matter the length, we just slice it
#             causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
#             attn_weights = attn_weights + causal_mask

#         # upcast attention to fp32
#         attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
#         attn_output = torch.matmul(attn_weights, value_states)

#         if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
#             raise ValueError(
#                 f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
#                 f" {attn_output.size()}"
#             )

#         attn_output = attn_output.transpose(1, 2).contiguous()

#         attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

#         if self.config.pretraining_tp > 1:
#             attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
#             o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
#             attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
#         else:
#             attn_output = self.o_proj(attn_output)

#         if not output_attentions:
#             attn_weights = None

#         return attn_output, attn_weights, past_key_value





class FusionLlamaConfig(LlamaConfig):
    """Configuration class for FusionLlama model"""
    model_type = "fusion_llama"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fusion-specific settings
        self.use_vision_fusion = kwargs.pop("use_vision_fusion", True)
        self.fusion_alpha = kwargs.pop("fusion_alpha", DEFAULT_FUSION_ALPHA)
        self.fusion_dropout = kwargs.pop("fusion_dropout", DEFAULT_FUSION_DROPOUT)
        self.fusion_targets = kwargs.pop("fusion_targets", DEFAULT_FUSION_TARGETS)
        self.mm_hidden_size = kwargs.pop("mm_hidden_size", DEFAULT_FUSION_VISION_DIM)
        self.fusion_hidden_dim = kwargs.pop("fusion_hidden_dim", DEFAULT_FUSION_HIDDEN_DIM)
        self.layer_ids_to_inject = kwargs.pop("layer_ids_to_inject", DEFAULT_FUSION_LAYER_IDS_TO_INJECT)
        self.stable_fusion = kwargs.pop("stable_fusion", False)
        self.use_mm_proj = kwargs.pop("use_mm_proj", True)
        # Image token settings
        if not hasattr(self, "image_token_id"):
            self.image_token_id = IMAGE_TOKEN_INDEX


class FusionLlamaModel(FusionMetaModel, LlamaModel):
    """Llama model with vision fusion capabilities"""
    config_class = FusionLlamaConfig
    
    def __init__(self, config):
        # Initialize LlamaModel first, following LlavaLlamaModel's pattern
        LlamaModel.__init__(self, config)
        # Then initialize our meta model
        FusionMetaModel.__init__(self, config)
        self.vision_hidden_states = None
        
        # Get layer_ids_to_inject from config instead of accessing it as an instance attribute
        layer_ids_to_inject = getattr(config, "layer_ids_to_inject", DEFAULT_FUSION_LAYER_IDS_TO_INJECT)
        # Convert layer_ids_to_inject to list of integers
        if isinstance(layer_ids_to_inject, str):
            self.layer_ids_to_inject = [int(layer_id) for layer_id in layer_ids_to_inject.split(',')]
        else:
            self.layer_ids_to_inject = layer_ids_to_inject
        
        # Add fusion capabilities to first layer attention projections
        if getattr(config, "use_vision_fusion", True):
            def fusion_wrapper_for_self_attn(self_attn, *args, **kwargs):
                return self._fusion_attention_forward(self_attn, *args, **kwargs)
            
            for layer_id in self.layer_ids_to_inject:
                # Replace with fusion-aware forward method
                self.layers[layer_id].self_attn.forward = types.MethodType(
                    fusion_wrapper_for_self_attn,
                    self.layers[layer_id].self_attn
                )
        #     # # Store original forward method
        #     # self._original_attention_forward = self.layers[0].self_attn.forward
        #     if getattr(config, "stable_fusion", False):
        #         for projection in fusion_targets:
        #             if hasattr(self.layers[0].self_attn, projection):
        #                 setattr(self.layers[0].self_attn, projection, StableFusionLayer(
        #                     getattr(self.layers[0].self_attn, projection),
        #                     vision_dim=getattr(config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
        #                     hidden_dim=getattr(config, "fusion_hidden_dim", DEFAULT_FUSION_HIDDEN_DIM),
        #                     alpha=getattr(config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
        #                     dropout=getattr(config, "fusion_dropout", DEFAULT_FUSION_DROPOUT)
        #                 ))
        #     else:
        #         for projection in fusion_targets:
        #             if hasattr(self.layers[0].self_attn, projection):
        #                 setattr(self.layers[0].self_attn, projection, CustomLoRAVisionFusionLayer(
        #                     getattr(self.layers[0].self_attn, projection),
        #                     vision_dim=getattr(config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
        #                     alpha=getattr(config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
        #                     dropout=getattr(config, "fusion_dropout", DEFAULT_FUSION_DROPOUT)
        #                 ))
            # # Apply fusion layers to projections
            # self.setup_fusion_modules(
            #     vision_dim=getattr(config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
            #     targets=getattr(config, "fusion_targets", DEFAULT_FUSION_TARGETS),
            #     alpha=getattr(config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
            #     dropout=getattr(config, "fusion_dropout", DEFAULT_FUSION_DROPOUT)
            # )

            # def fusion_wrapper_for_self_attn(self_attn, *args, **kwargs):
            #     return self._fusion_attention_forward(self_attn, *args, **kwargs)

            # # Replace with fusion-aware forward method
            # self.layers[0].self_attn.forward = types.MethodType(
            #     fusion_wrapper_for_self_attn,
            #     self.layers[0].self_attn
            # )

            # self.setup_fusion_modules(
            #     vision_dim=getattr(config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
            #     targets=getattr(config, "fusion_targets", DEFAULT_FUSION_TARGETS),
            #     alpha=getattr(config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
            #     dropout=getattr(config, "fusion_dropout", DEFAULT_FUSION_DROPOUT)
            # )
            
            
    
    def _fusion_attention_forward(
        self,
        attn_self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config._attn_implementation == "flash_attention_2":
            output_attentions = False

            bsz, q_len, _ = hidden_states.size()
            if self.vision_hidden_states is not None:
                query_states = attn_self.q_proj(hidden_states, self.vision_hidden_states.clone())
                key_states = attn_self.k_proj(hidden_states, self.vision_hidden_states.clone())
                value_states = attn_self.v_proj(hidden_states, self.vision_hidden_states.clone())
            else:
                query_states = attn_self.q_proj(hidden_states)
                key_states = attn_self.k_proj(hidden_states)
                value_states = attn_self.v_proj(hidden_states)

            # Flash attention requires the input to have the shape
            # batch_size x seq_length x head_dim x hidden_dim
            # therefore we just need to keep the original shape
            query_states = query_states.view(bsz, q_len, attn_self.num_heads, attn_self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn_self.num_key_value_heads, attn_self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn_self.num_key_value_heads, attn_self.head_dim).transpose(1, 2)

            cos, sin = attn_self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, attn_self.layer_idx, cache_kwargs)

            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

            dropout_rate = attn_self.attention_dropout if attn_self.training else 0.0

            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in the correct dtype just to be sure everything works as expected.
            # This might slowdown training & inference so it is recommended to not cast the LayerNorms
            # in fp32. (LlamaRMSNorm handles it correctly)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                # Handle the case where the model is quantized
                elif hasattr(attn_self.config, "_pre_quantization_dtype"):
                    target_dtype = attn_self.config._pre_quantization_dtype
                else:
                    target_dtype = attn_self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            attn_output = attn_self._flash_attention_forward(
                query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
            )

            attn_output = attn_output.reshape(bsz, q_len, attn_self.hidden_size).contiguous()
            attn_output = attn_self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        else:
            if attn_self.config.pretraining_tp > 1:
            ### NEED TO IMPLEMENT FUSION FOR MULTI-GPU TRAINING

                key_value_slicing = (attn_self.num_key_value_heads * attn_self.head_dim) // attn_self.config.pretraining_tp
                query_slices = attn_self.q_proj.weight.split(
                    (attn_self.num_heads * attn_self.head_dim) // attn_self.config.pretraining_tp, dim=0
                )
                key_slices = attn_self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = attn_self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(attn_self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(attn_self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(attn_self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

            else:
                if self.vision_hidden_states is not None:
                    query_states = attn_self.q_proj(hidden_states, self.vision_hidden_states.clone())
                    key_states = attn_self.k_proj(hidden_states, self.vision_hidden_states.clone())
                    value_states = attn_self.v_proj(hidden_states, self.vision_hidden_states.clone())
                else:
                    query_states = attn_self.q_proj(hidden_states)
                    key_states = attn_self.k_proj(hidden_states)
                    value_states = attn_self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attn_self.num_heads, attn_self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn_self.num_key_value_heads, attn_self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn_self.num_key_value_heads, attn_self.head_dim).transpose(1, 2)

            cos, sin = attn_self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, attn_self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, attn_self.num_key_value_groups)
            value_states = repeat_kv(value_states, attn_self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=attn_self.attention_dropout, training=attn_self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, attn_self.num_heads, q_len, attn_self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, attn_self.num_heads, q_len, attn_self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            attn_output = attn_output.reshape(bsz, q_len, attn_self.hidden_size)

            if attn_self.config.pretraining_tp > 1:
                attn_output = attn_output.split(attn_self.hidden_size // attn_self.config.pretraining_tp, dim=2)
                o_proj_slices = attn_self.o_proj.weight.split(attn_self.hidden_size // attn_self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(attn_self.config.pretraining_tp)])
            else:
                attn_output = attn_self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

    def setup_fusion_modules(self, vision_dim, targets=["q", "k", "v"], alpha=8.0, dropout=0.1):
        """Set up fusion modules for the specified projection targets"""
        if not hasattr(self, 'layers') or len(self.layers) == 0:
            return
        
        # # Convert string target to list if needed
        # if isinstance(targets, str):
        #     targets = targets.split(',')
        
        # Get first layer's attention module
        attn = self.layers[0].self_attn
        
        # # Set up fusion modules for each target
        # vision_dim = getattr(self.config, 'mm_hidden_size', DEFAULT_FUSION_VISION_DIM)
        
        for target in targets:
            target_name = f"{target}_proj"
            if hasattr(attn, target_name):
                # Get the original projection layer
                original_proj = getattr(attn, target_name)
                
                # Replace with fusion-enabled layer
                fusion_layer = CustomLoRAVisionFusionLayer(
                    original_proj, 
                    vision_dim=vision_dim,
                    alpha=alpha,
                    dropout=dropout
                )
                
                # Set the new layer
                setattr(attn, target_name, fusion_layer)



class FusionLlamaForCausalLM(LlamaForCausalLM, FusionMetaForCausalLM):
    """Causal language model with vision fusion capabilities"""
    config_class = FusionLlamaConfig
    
    def __init__(self, config):
        # Follow pattern from LlavaLlamaForCausalLM
        LlamaForCausalLM.__init__(self, config)
        
        # Replace the model with our fusion model
        self.model = FusionLlamaModel(config)
        
        # Initialize vocabulary-related components
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if not getattr(config, "load_from_lora", True):
           print("="*100)
           print("Using the fusion modules creation function in the FusionLlamaForCausalLM class")
           print("="*100)
           self._create_fusion_modules(config)
        # Initialize weights
        self.post_init()

    def _create_fusion_modules(self, config):
        # Parse targets
        fusion_targets = getattr(config, "fusion_targets", DEFAULT_FUSION_TARGETS)
        if isinstance(fusion_targets, str):
            fusion_targets = fusion_targets.split(',')
        layer_ids_to_inject = getattr(config, "layer_ids_to_inject", DEFAULT_FUSION_LAYER_IDS_TO_INJECT)
        if isinstance(layer_ids_to_inject, str):
            layer_ids_to_inject = [int(layer_id) for layer_id in layer_ids_to_inject.split(',')]
        # Add fusion layers to each target
        for layer_id in layer_ids_to_inject:
            attn = self.model.layers[layer_id].self_attn
            for target in fusion_targets:
                proj_name = f"{target}_proj"
                if hasattr(attn, proj_name):
                    original_proj = getattr(attn, proj_name)
                    
                    # Choose the appropriate fusion layer type
                    if getattr(config, "use_stable_fusion", False):
                        fusion_layer = StableFusionLayer(
                            original_proj,
                            vision_dim=getattr(config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
                            alpha=getattr(config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
                            dropout=getattr(config, "fusion_dropout", DEFAULT_FUSION_DROPOUT),
                            fusion_hidden_dim=getattr(config, "fusion_hidden_dim", DEFAULT_FUSION_HIDDEN_DIM)
                        )
                    else:
                        fusion_layer = CustomLoRAVisionFusionLayer(
                            original_proj,
                            vision_dim=getattr(config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
                            alpha=getattr(config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
                            dropout=getattr(config, "fusion_dropout", DEFAULT_FUSION_DROPOUT)
                        )
                        
                    # Replace the projection with fusion layer
                    setattr(attn, proj_name, fusion_layer)
    
    def get_model(self):
        """Return the underlying model"""
        return self.model
   

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass that handles both text and vision inputs
        """
        self.model.vision_hidden_states = None

        # Process inputs when we need inputs_embeds or vision_embeds
        if inputs_embeds is None or (vision_embeds is None and images is not None):
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                computed_vision_embeds
            ) = self.prepare_inputs_labels_for_fusion(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            
            # Use computed vision_embeds if available and none was provided
            if vision_embeds is None and computed_vision_embeds is not None:
                vision_embeds = computed_vision_embeds.clone()
            # print("="*100)
            # print("IN the forward function, and the vision embeds should be prepared")
            # print("="*100)
        # print("="*100)
        # print("IN the forward function, and the vision embeds are not None")
        # print(f"In the forward function, the images are not None: {images is not None}")
        # print("="*100)

        # Pad the vision_embeds to the same length as the inputs_embeds
        # print("="*100)
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # Store vision_embeds in the model for the attention layer to access
        self.model.vision_hidden_states = vision_embeds
        # Change the dtype of the vision_embeds to the same as the inputs_embeds

        # Call the parent's forward without vision_embeds, similar to LlavaLlamaForCausalLM
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
    
    @torch.no_grad()
    def generate_advanced(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """Generate text with optional image context"""
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        vision_embeds = None
        # Process images if provided
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                computed_vision_embeds
            ) = self.prepare_inputs_labels_for_fusion(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes
            )
            # print("="*100)
            # print("IN the generate function, and the vision embeds are not None")
            # print("="*100)

            # Use the computed vision embeds
            vision_embeds = computed_vision_embeds
        else:
            # Just convert input ids to embeddings
            inputs_embeds = self.model.embed_tokens(inputs)

        # Store vision_embeds in the model
        self.model.vision_hidden_states = vision_embeds
        # print("="*100)
        # print("IN the generate function, and the vision embeds are None")
        # print("="*100)
        # print(f"Shape of the vision embeds: {vision_embeds.shape}")
        # print("="*100)
        # print(f"Shape of the inputs_embeds: {inputs_embeds.shape}")
        # print("="*100)

        # Call generate without vision_embeds
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None,
        inputs_embeds=None, 
        **kwargs
    ):
        """Prepare inputs for generation with vision fusion support"""
        print("="*100)
        print("IN the prepare_inputs_for_generation function")
        print("="*100)
        # Extract vision embeddings from kwargs if they were passed
        # vision_embeds = kwargs.pop("vision_embeds", None)
        # # If no vision_embeds in kwargs but we have them in the model, preserve them
        # if vision_embeds is None and hasattr(self.model, 'vision_hidden_states') and self.model.vision_hidden_states is not None:
        #     vision_embeds = self.model.vision_hidden_states
        # print("="*100)
        # print(f"In the prepare_inputs_for_generation function, the vision embeds are not None: {vision_embeds is not None}")
        # print("="*100)
        # Call parent implementation to get the standard inputs
        inputs = super().prepare_inputs_for_generation(
            input_ids, 
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            **kwargs
        )

        # Also pass vision_embeds in the inputs dict so it's available for the next step
        # inputs['vision_embeds'] = vision_embeds

        # Keep images and image_sizes consistent with our existing code
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        
        # Always pass vision_embeds
        vision_embeds = kwargs.get("vision_embeds", None)
        if vision_embeds is None and hasattr(self.model, 'vision_hidden_states'):
            vision_embeds = self.model.vision_hidden_states
            print("="*100)
            print("IN the prepare_inputs_for_generation function, and the vision embeds are not None")
            print("="*100)
    
        # # IF inputs has a key called inputs_embeds, then we need to pad the vision_embeds to the same length as the inputs_embeds
        # if 'inputs_embeds' in inputs and vision_embeds is not None:
        #     vision_embeds = torch.cat([
        #         vision_embeds, 
        #         torch.zeros(
        #             vision_embeds.shape[0],                      # batch size 
        #             inputs['inputs_embeds'].shape[1] - vision_embeds.shape[1],  # padding length
        #             vision_embeds.shape[-1],                          # embedding dimension
        #             device=vision_embeds.device, 
        #             dtype=vision_embeds.dtype
        #         )
        #     ], dim=1)  # dim=1 is the sequence length dimension
        #     print(f"Shape of the vision embeds after padding: {vision_embeds.shape}")
        #     print("="*100)
        #     print(f"Shape of the inputs_embeds: {inputs['inputs_embeds'].shape}")
        #     print("="*100)
        #     self.model.vision_hidden_states = vision_embeds
        # inputs['vision_embeds'] = vision_embeds

        # # Critical: force recalculation of all attention for accurate fusion
        # # This is slower but ensures vision-text integration is precise
        # inputs['past_key_values'] = None
        # inputs['use_cache'] = False
        
        return inputs
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        max_new_tokens: int = 100,
        **kwargs
    ) -> torch.LongTensor:
        """
        Simple greedy autoregressive generation without using cache.
        This is useful when custom fusion mechanisms don't work well with KV caching.
        """
        # position_ids = kwargs.pop("position_ids", None)
        # attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("use_cache", None)

        # print("="*100)
        # print(f"Input ids: {input_ids}")
        # print("="*100)
        # # vision_embeds = None
        # # # Process images if provided
        # if images is not None:
        #     (
        #         input_ids,
        #         position_ids,
        #         attention_mask,
        #         _,
        #         _,
        #         _,
        #         computed_vision_embeds
        #     ) = self.prepare_inputs_labels_for_fusion(
        #         input_ids,
        #         None,
        #         None,
        #         None,
        #         None,
        #         images,
        #         image_sizes
        #     )
        #     # Use the computed vision embeds
        #     vision_embeds = computed_vision_embeds
        
        # # # Store vision_embeds in the model
        # self.model.vision_hidden_states = vision_embeds
        
        # # Ensure we're operating on input_ids
        # if inputs_embeds is not None:
        #     raise ValueError("Simple generate doesn't support inputs_embeds yet")
        
        
        
        # # Create dynamic attention_mask if not provided
        # if attention_mask is None:
        #     attention_mask = torch.ones_like(input_ids)
        
        # # Create dynamic position_ids if not provided
        # if position_ids is None:
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        
        # Initialize with the input sequence
        generated_ids = input_ids.clone()
        
        # Perform generation step by step (without caching)
        for _ in range(max_new_tokens):
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                # Forward pass through the model
                outputs = self.forward(
                    input_ids=generated_ids,
                    images=images,
                    image_sizes=image_sizes,
                    use_cache=False,  # Critical: don't use caching
                    return_dict=True,
                    # **kwargs
                )
            
            # Get the next token logits (last position)
            # print("="*100)
            # print(f"Logits shape: {outputs.logits.shape}")
            # print("="*100)
            next_token_logits = outputs.logits[:, -1, :]
            # print("="*100)
            # print(f"Next token logits shape: {next_token_logits.shape}")
            # print("="*100)
            # Simple greedy search: pick the token with the highest probability
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            # print("="*100)
            # print(f"Maximum next tokens logits: {next_token_logits.max()}")
            # print("="*100)
            # print(f"Next tokens: {next_tokens}")
            # print("="*100)
            # Append the new tokens to the current sequence
            generated_ids = torch.cat([generated_ids, next_tokens], dim=-1)

            # print("="*100)
            # print(f"Generated ids shape: {generated_ids.shape}")
            # print("="*100)
            
            # # Update position_ids for the new token position
            # curr_length = generated_ids.shape[1]
            # new_position_ids = torch.cat([
            #     position_ids,
            #     torch.ones(position_ids.shape[0], 1, device=generated_ids.device, dtype=torch.long) * curr_length
            # ], dim=1)
            
            # # Update attention mask for the new token
            # new_attention_mask = torch.cat([
            #     attention_mask,
            #     torch.ones(attention_mask.shape[0], 1, device=generated_ids.device)
            # ], dim=1)

            # # Update position_ids and attention_mask for next iteration
            # position_ids = new_position_ids
            # attention_mask = new_attention_mask
            
            eos_token_id = self.model.config.eos_token_id
            # print("="*100)
            # print(f"Eos token id: {eos_token_id}")
            # print("="*100)
            
            # print(f"Next tokens: {next_tokens}")
            # print("="*100)
            if next_tokens == eos_token_id:
                break
            # Optional: check for EOS token and stop if all sequences have ended
            # (uncomment and modify as needed)
            # eos_token_id = kwargs.get("eos_token_id", None)
            # if eos_token_id is not None and (next_tokens == eos_token_id).all():
            #     break
            # print("="*100)
            # print(f"Generated ids: {generated_ids}")
            # print("="*100)
        # Remove the inputs ids from the generated ids
        generated_ids = generated_ids[:, input_ids.shape[1]:]
        return generated_ids





# Register with AutoConfig and AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("fusion_llama", FusionLlamaConfig)
AutoModelForCausalLM.register(FusionLlamaConfig, FusionLlamaForCausalLM)