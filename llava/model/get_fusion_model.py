# import torch
from llava.constants import DEFAULT_FUSION_TARGETS, DEFAULT_FUSION_ALPHA, DEFAULT_FUSION_DROPOUT, DEFAULT_FUSION_HIDDEN_DIM, DEFAULT_FUSION_VISION_DIM, DEFAULT_FUSION_LAYER_IDS_TO_INJECT
from llava.model.fusion_layer import CustomLoRAVisionFusionLayer, StableFusionLayer, ModalityAwareFusionLayer

def get_fusion_model(model, 
                    vision_dim=DEFAULT_FUSION_VISION_DIM, 
                    fusion_targets=DEFAULT_FUSION_TARGETS,
                    alpha=DEFAULT_FUSION_ALPHA, 
                    dropout=DEFAULT_FUSION_DROPOUT,
                    stable_fusion=False,
                    fusion_hidden_dim=DEFAULT_FUSION_HIDDEN_DIM,
                    layer_ids_to_inject=DEFAULT_FUSION_LAYER_IDS_TO_INJECT):
    """
    Add fusion layers to a pretrained model - similar to PEFT's get_peft_model
    
    Args:
        model: A FusionLlamaForCausalLM model that's already initialized with pretrained weights
        vision_dim: Dimension of vision embeddings
        fusion_targets: Comma-separated list of target projections ('q', 'k', 'v')
        alpha: Scaling factor for fusion output
        dropout: Dropout probability for fusion layers
        stable_fusion: Whether to use StableFusionLayer instead of CustomLoRAVisionFusionLayer
    
    Returns:
        The same model with fusion layers added to specified projections
    """
    # Ensure we have a valid model
    if not hasattr(model, 'model') or not hasattr(model.model, 'layers'):
        raise ValueError("Model must be a FusionLlamaForCausalLM model")
    
    # Convert layer_ids_to_inject to a list of integers
    layer_ids_to_inject = [int(id) for id in layer_ids_to_inject.split(',')]

    for layer_id in layer_ids_to_inject:
        # Get the first layer's attention module
        attn = model.model.layers[layer_id].self_attn
        
        # Parse targets
        if isinstance(fusion_targets, str):
            fusion_targets = fusion_targets.split(',')
        
        # Add fusion layers to each target
        for target in fusion_targets:
            proj_name = f"{target}_proj"
            if hasattr(attn, proj_name):
                original_proj = getattr(attn, proj_name)
                
                # Choose the appropriate fusion layer type
                if stable_fusion:
                    # fusion_layer = StableFusionLayer(
                    #     original_proj,
                    #     vision_dim=vision_dim,
                    #     alpha=alpha,
                    #     dropout=dropout,
                    #     fusion_hidden_dim=fusion_hidden_dim
                    # )
                    fusion_layer = ModalityAwareFusionLayer(
                        original_proj,
                        vision_dim=vision_dim,
                        alpha=alpha,
                        dropout=dropout
                    )
                else:
                    fusion_layer = CustomLoRAVisionFusionLayer(
                        original_proj,
                        vision_dim=vision_dim,
                        alpha=alpha,
                        dropout=dropout
                    )
                    
                # Replace the projection with fusion layer
                setattr(attn, proj_name, fusion_layer)
    
    return model 