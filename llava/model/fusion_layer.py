import torch
import torch.nn as nn
import math
from llava.constants import DEFAULT_FUSION_ALPHA, DEFAULT_FUSION_DROPOUT, DEFAULT_FUSION_HIDDEN_DIM
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm

# class CustomLoRAVisionFusionLayer(nn.Module):
#     """Custom LoRA-like layer that fuses vision embeddings with language features"""
#     def __init__(self, original_layer, vision_dim, alpha=DEFAULT_FUSION_ALPHA, dropout=DEFAULT_FUSION_DROPOUT):
#         super().__init__()
        
#         self.compute_dtype = original_layer.weight.dtype
#         print(f"compute_dtype: {self.compute_dtype}")

#         self.original_weight = nn.Parameter(original_layer.weight.detach().clone(), requires_grad=False)
#         if hasattr(original_layer, 'bias') and original_layer.bias is not None:
#             self.original_bias = nn.Parameter(original_layer.bias.detach().clone(), requires_grad=False)
#         else:
#             self.register_parameter('original_bias', None)
            
#         self.in_features = original_layer.in_features
#         self.out_features = original_layer.out_features
#         self.vision_dim = vision_dim

#         # Create LoRA matrices
#         self.rank = vision_dim
#         std_dev = 1 / math.sqrt(self.rank)
#         self.up_A = nn.Parameter(torch.randn(self.in_features, self.rank, dtype=self.compute_dtype) * std_dev)
#         # self.down_B = nn.Parameter(torch.zeros(2 * self.rank, self.out_features, dtype=self.compute_dtype))
#         self.down_B = nn.Parameter(torch.zeros(self.rank, self.out_features, dtype=self.compute_dtype))

#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.compute_dtype))
#         self.vision_norm = LlamaRMSNorm(self.vision_dim)
        
#     def forward(self, x, vision_embeds=None):
        
#         original_output = F.linear(x, self.original_weight, self.original_bias)
        
#         up_A_out = x @ self.up_A
#         vision_embeds = self.vision_norm(vision_embeds)
        
#         fusion = up_A_out + vision_embeds
#         lora_output = fusion @ self.down_B
        
#         # Add LoRA output to original output
#         output = original_output + self.alpha * lora_output
        

#         return output



# class CustomLoRAVisionFusionLayer(nn.Module):
#     """Custom LoRA-like layer that fuses vision embeddings with language features"""
#     def __init__(self, original_layer, vision_dim, alpha=DEFAULT_FUSION_ALPHA, dropout=DEFAULT_FUSION_DROPOUT):
#         super().__init__()
#         self.compute_dtype = original_layer.weight.dtype
#         print(f"compute_dtype: {self.compute_dtype}")

#         # Ensure original weights are in compute_dtype
#         self.original_weight = nn.Parameter(original_layer.weight.detach().clone().to(self.compute_dtype), requires_grad=False)
#         if hasattr(original_layer, 'bias') and original_layer.bias is not None:
#             self.original_bias = nn.Parameter(original_layer.bias.detach().clone().to(self.compute_dtype), requires_grad=False)
#         else:
#             self.register_parameter('original_bias', None)
            
#         self.in_features = original_layer.in_features
#         self.out_features = original_layer.out_features
#         self.vision_dim = vision_dim

#         # Create LoRA matrices with explicit dtype
#         self.rank = vision_dim
#         std_dev = 1 / math.sqrt(self.rank)
#         self.up_A = nn.Parameter(torch.randn(self.in_features, self.rank, dtype=self.compute_dtype) * std_dev)
#         self.down_B = nn.Parameter(torch.randn(2* self.rank, self.out_features, dtype=self.compute_dtype) * 0.0001)
        
#         # Ensure alpha is in compute_dtype
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.compute_dtype))
#         self.vision_norm = LlamaRMSNorm(self.vision_dim)
        
#     def forward(self, x, vision_embeds=None):
#         # Ensure input is in compute_dtype
#         x = x.to(self.compute_dtype)
        
#         # Return early if no vision embeds
#         original_output = F.linear(x, self.original_weight, self.original_bias)

#         if vision_embeds is None or not isinstance(vision_embeds, torch.Tensor):
#             vision_embeds = torch.zeros(x.shape[:-1] + (self.rank,), 
#                                        device=x.device, 
#                                        dtype=self.compute_dtype)
#         else:
#             # Ensure vision_embeds is in compute_dtype
#             vision_embeds = vision_embeds.to(self.compute_dtype)

#         # Project and normalize
#         up_A_out = x @ self.up_A
#         vision_embeds = self.vision_norm(vision_embeds)
        
#         # Concatenate and project
#         fusion = torch.cat([up_A_out, vision_embeds], dim=-1)
#         lora_output = fusion @ self.down_B
        
#         # Add LoRA output to original output
#         output = original_output + self.alpha * lora_output
        
#         return output


class CustomLoRAVisionFusionLayer(nn.Module):
    """Custom LoRA-like layer that fuses vision embeddings with language features"""
    def __init__(self, original_layer, vision_dim, alpha=DEFAULT_FUSION_ALPHA, dropout=DEFAULT_FUSION_DROPOUT):
        super().__init__()
        self.compute_dtype = original_layer.weight.dtype
        print(f"compute_dtype: {self.compute_dtype}")

        # Ensure original weights are in compute_dtype
        self.original_weight = nn.Parameter(original_layer.weight.detach().clone().to(self.compute_dtype), requires_grad=False)
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.original_bias = nn.Parameter(original_layer.bias.detach().clone().to(self.compute_dtype), requires_grad=False)
        else:
            self.register_parameter('original_bias', None)
            
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.vision_dim = vision_dim
        self.projection_1 = nn.Linear(self.vision_dim, self.out_features)
        self.gelu = nn.GELU()
        self.projection_2 = nn.Linear(self.out_features, self.out_features)
        # Ensure alpha is in compute_dtype
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.compute_dtype))
        
    def forward(self, x, vision_embeds=None):
        # Ensure input is in compute_dtype
        x = x.to(self.compute_dtype)
        
        # Return early if no vision embeds
        original_output = F.linear(x, self.original_weight, self.original_bias)

        if vision_embeds is None or not isinstance(vision_embeds, torch.Tensor):
            vision_embeds = torch.zeros(x.shape[:-1] + (self.rank,), 
                                       device=x.device, 
                                       dtype=self.compute_dtype)
        else:
            # Ensure vision_embeds is in compute_dtype
            vision_embeds = vision_embeds.to(self.compute_dtype)

        # Vision output
        vision_output = self.projection_1(vision_embeds)
        vision_output = self.gelu(vision_output)
        vision_output = self.projection_2(vision_output)
        # Add to original output
        output = original_output + self.alpha * vision_output
        
        return output

class ModalityAwareFusionLayer(nn.Module):
    """Fusion layer that handles modalities without padding zeros"""
    def __init__(self, original_layer, vision_dim, alpha=DEFAULT_FUSION_ALPHA, dropout=DEFAULT_FUSION_DROPOUT):
        super().__init__()
        
        # Copy original weights like CustomLoRAVisionFusionLayer
        self.compute_dtype = original_layer.weight.dtype
        
        self.original_weight = nn.Parameter(original_layer.weight.detach().clone(), requires_grad=False)
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.original_bias = nn.Parameter(original_layer.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter('original_bias', None)
            
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.vision_dim = vision_dim
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=self.compute_dtype))
        
        # Initialize weights with controlled randomness
        std_dev = 1 / math.sqrt(vision_dim)
        self.up_A = nn.Parameter(torch.randn(self.in_features, vision_dim, dtype=self.compute_dtype) * std_dev)
        
        # Separate projection matrices for text and vision
        self.text_down = nn.Parameter(torch.randn(vision_dim, self.out_features, dtype=self.compute_dtype) * 0.01)
        self.vision_down = nn.Parameter(torch.randn(vision_dim, self.out_features, dtype=self.compute_dtype) * 0.01)
        
        # Vision normalization
        self.vision_norm = LlamaRMSNorm(vision_dim)
        
    def forward(self, x, vision_embeds=None):
        # Original output
        original_output = F.linear(x, self.original_weight, self.original_bias)
        
        # Return early if no vision embeds
        if vision_embeds is None or not isinstance(vision_embeds, torch.Tensor):
            # Still process text features
            up_A_out = x @ self.up_A
            text_contribution = up_A_out @ self.text_down
            output = original_output + self.alpha * text_contribution
            return output
        
        # Make sure the vision_embeds is the same type as x
        vision_embeds = vision_embeds.to(x.dtype)
        
        # Project text features
        up_A_out = x @ self.up_A
        text_contribution = up_A_out @ self.text_down
        
        # Process vision features
        vision_embeds = self.vision_norm(vision_embeds)
        vision_contribution = vision_embeds @ self.vision_down
        
        # Add contributions directly (no concatenation needed)
        lora_output = text_contribution + vision_contribution
        
        # Add to original output
        output = original_output + self.alpha * lora_output
        
        return output


class StableFusionLayer(nn.Module):
    """Stable vision-language fusion layer with proper initialization and scaling"""
    def __init__(self, original_layer, vision_dim, hidden_dim=DEFAULT_FUSION_HIDDEN_DIM, alpha=DEFAULT_FUSION_ALPHA, dropout=DEFAULT_FUSION_DROPOUT):
        super().__init__()
        # Get original weights
        original_weight = original_layer.weight.clone()
        self.original_weight = nn.Parameter(original_weight)
        
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.clone())
        else:
            self.register_parameter('bias', None)
            
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.vision_dim = vision_dim
        
        dtype = original_weight.dtype

        # More stable initialization with intermediate dimensionality
        self.text_down = nn.Parameter(torch.randn(self.in_features, hidden_dim, dtype=dtype) * 
                                     (1.0 / math.sqrt(self.in_features)))
        
        self.vision_down = nn.Parameter(torch.randn(vision_dim, hidden_dim, dtype=dtype) * 
                                       (1.0 / math.sqrt(vision_dim)))
        
        # IMPORTANT: Initialize with small non-zero values
        self.fusion_up = nn.Parameter(torch.randn(2 * hidden_dim, self.out_features, dtype=dtype) * 
                                     (0.01 / math.sqrt(2 * hidden_dim)))  # Small but non-zero
        
        # Add layer normalization for better feature alignment
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.vision_norm = nn.LayerNorm(hidden_dim)
        
        # Gating mechanism for controlled integration
        self.gate_bias = -2.0  # Start with low vision influence (sigmoid(-2) ≈ 0.12)
        self.vision_gate = nn.Parameter(torch.full((1, self.out_features), self.gate_bias, dtype=dtype))
        
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, vision_embeds=None):

        # Original output
        original_output = F.linear(x, self.original_weight, self.bias)

        # Return early if no vision embeds
        if vision_embeds is None or not isinstance(vision_embeds, torch.Tensor):
            return original_output
    
        
        # Force the types of vision_embeds to match x
        vision_embeds = vision_embeds.to(x.dtype)
        
        # Process text features
        x_dropped = self.dropout(x)
        text_proj = x_dropped @ self.text_down
        text_proj = self.text_norm(text_proj)
        
        # Process vision features
        vision_proj = vision_embeds @ self.vision_down
        vision_proj = self.vision_norm(vision_proj)
        
        # Combine features
        fusion = torch.cat([text_proj, vision_proj], dim=-1)
        
        # Project to output space
        fusion_output = fusion @ self.fusion_up
        
        # Apply gating to control vision influence
        vision_gate = torch.sigmoid(self.vision_gate)
        
        # Add gated fusion output to original output
        output = original_output + self.alpha * fusion_output * vision_gate
        
        return output
    
    
# class StableFusionLayer2(nn.Module):
#     """Stable vision-language fusion layer with proper initialization and scaling"""
#     def __init__(self, original_layer, vision_dim, hidden_dim=DEFAULT_FUSION_HIDDEN_DIM, alpha=DEFAULT_FUSION_ALPHA, dropout=DEFAULT_FUSION_DROPOUT):
#         super().__init__()
#         # Get original weights
#         original_weight = original_layer.weight.clone()
#         self.original_weight_fusion = nn.Parameter(original_weight)
        
#         if hasattr(original_layer, 'bias') and original_layer.bias is not None:
#             self.bias_fusion = nn.Parameter(original_layer.bias.clone())
#         else:
#             self.register_parameter('bias', None)
            
#         self.in_features = original_layer.in_features
#         self.out_features = original_layer.out_features
#         self.vision_dim = vision_dim
        
#         dtype = original_weight.dtype

#         # More stable initialization with intermediate dimensionality
#         self.text_down_fusion = nn.Parameter(torch.randn(self.in_features, hidden_dim, dtype=dtype) * 
#                                      (1.0 / math.sqrt(self.in_features)))
        
#         self.vision_down_fusion = nn.Parameter(torch.randn(vision_dim, hidden_dim, dtype=dtype) * 
#                                        (1.0 / math.sqrt(vision_dim)))
        
#         # IMPORTANT: Initialize with small non-zero values
#         self.fusion_up_fusion = nn.Parameter(torch.randn(2 * hidden_dim, self.out_features, dtype=dtype) * 
#                                      (0.01 / math.sqrt(2 * hidden_dim)))  # Small but non-zero
        
#         # Add layer normalization for better feature alignment
#         self.text_norm_fusion = nn.LayerNorm(hidden_dim)
#         self.vision_norm_fusion = nn.LayerNorm(hidden_dim)
        
#         # Gating mechanism for controlled integration
#         self.gate_bias_fusion = -2.0  # Start with low vision influence (sigmoid(-2) ≈ 0.12)
#         self.vision_gate_fusion = nn.Parameter(torch.full((1, self.out_features), self.gate_bias_fusion, dtype=dtype))
        
#         self.alpha = alpha
#         self.dropout_fusion = nn.Dropout(p=dropout)
        
#     def forward(self, x, vision_embeds=None):
#         # Original output
#         original_output = F.linear(x, self.original_weight_fusion, self.bias_fusion)
        
#         # Return early if no vision embeds
#         if vision_embeds is None or not isinstance(vision_embeds, torch.Tensor):
#             return original_output
        
#         # Force the types of vision_embeds to match x
#         vision_embeds = vision_embeds.to(x.dtype)
        
#         # Process text features
#         x_dropped = self.dropout_fusion(x)
#         text_proj = x_dropped @ self.text_down_fusion
#         text_proj = self.text_norm_fusion(text_proj)
        
#         # Process vision features
#         vision_proj = vision_embeds @ self.vision_down_fusion
#         vision_proj = self.vision_norm_fusion(vision_proj)
        
#         # Combine features
#         fusion = torch.cat([text_proj, vision_proj], dim=-1)
        
#         # Project to output space
#         fusion_output = fusion @ self.fusion_up_fusion
        
#         # Apply gating to control vision influence
#         vision_gate = torch.sigmoid(self.vision_gate_fusion)
        
#         # Add gated fusion output to original output
#         output = original_output + self.alpha * fusion_output * vision_gate
        
#         return output