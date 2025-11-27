import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class DeepLearningRegistrationPipeline(nn.Module):
    """
    Wrapper for KeyMorph-based multi-modal alignment.
    References: "KeyMorph: Robust Multi-modal Affine Registration via Unsupervised Keypoint Detection"
    This version is adapted for a generic, dictionary-based input.
    """
    def __init__(self, keymorph_network: nn.Module, fixed_modality: str, moving_modalities: List[str], output_order: List[str]):
        super().__init__()
        if keymorph_network is None:
            raise ValueError("A keymorph_network must be provided for registration.")
        self.keymorph = keymorph_network
        self.fixed_modality = fixed_modality
        self.moving_modalities = moving_modalities
        self.output_order = output_order # Defines the channel order of the output tensor

    def _stn(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Spatial Transformer Network function to warp the input tensor x using the affine matrix theta."""
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs registration and fusion.
        Args:
            inputs: A dictionary where keys are modality names (e.g., 'S1', 'S2')
                    and values are corresponding tensors of shape (B, C, H, W).
        Returns:
            A single fused tensor.
        """
        if self.fixed_modality not in inputs:
            raise ValueError(f"Fixed modality '{self.fixed_modality}' not found in input dictionary.")
        
        fixed_tensor = inputs[self.fixed_modality]
        kp_fixed = self.keymorph(fixed_tensor)

        aligned_tensors = {self.fixed_modality: fixed_tensor}

        # 1. Align all "moving" modalities to the "fixed" one.
        for modality in self.moving_modalities:
            if modality not in inputs:
                # Or raise an error, depending on desired strictness
                print(f"Warning: Moving modality '{modality}' not in inputs. Skipping.")
                continue

            moving_tensor = inputs[modality]
            kp_moving = self.keymorph(moving_tensor)
            theta = self.compute_affine_matrix(kp_moving, kp_fixed) # Align moving -> fixed
            aligned_tensors[modality] = self._stn(moving_tensor, theta)

        # 2. Concatenate all tensors in the user-specified output order.
        tensors_to_fuse = [aligned_tensors[mod] for mod in self.output_order if mod in aligned_tensors]
        fused_tensor = torch.cat(tensors_to_fuse, dim=1)

        return fused_tensor

    def compute_affine_matrix(self, source_kp: torch.Tensor, target_kp: torch.Tensor) -> torch.Tensor:
        # Placeholder for the actual keypoint-based matrix computation.
        # For now, it returns an identity matrix, resulting in no transformation.
        B = source_kp.shape[0]
        theta = torch.eye(2, 3, device=source_kp.device).unsqueeze(0).repeat(B, 1, 1)
        return theta