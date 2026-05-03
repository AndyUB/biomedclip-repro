import torch
import torch.nn as nn


class BiomedCLIPVisualWrapper(nn.Module):
    """Wraps BiomedCLIP's open_clip visual encoder to match METER's vit_model interface.

    Returns (B, 197, 768) all patch tokens (including CLS) for 224x224 input,
    matching the output format of METER's CLIP VisualTransformer.
    """

    MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(self):
        super().__init__()
        import open_clip
        clip_model, _ = open_clip.create_model_from_pretrained(self.MODEL_NAME)
        # Detach the trunk so its parameters are tracked by this module
        self.trunk = clip_model.visual.trunk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns (B, N+1, D) = (B, 197, 768) for 224x224 with patch_size=16
        return self.trunk.forward_features(x)
