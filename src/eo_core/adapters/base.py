from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch.nn as nn

class BaseAdapter(ABC):
    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the adapter with its specific parameters from the config.
        
        Args:
            params: Dictionary of parameters specific to this adapter/model 
                    (e.g., num_classes, image_size, specific_paths).
        """
        self.params = params

    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        Constructs and returns the PyTorch model instance.
        This is where you instantiate your specific architecture (e.g., ResNet, UNet).
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """
        Takes raw input and prepares it for the model.
        
        Args:
            raw_input: Input data provided by the caller (e.g., file path, dictionary of params).
            
        Returns:
            Tensor or batch of tensors ready for model consumption.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """
        Takes raw model output and converts it to a user-friendly format.
        
        Args:
            model_output: Raw output from the model (e.g., logits).
            
        Returns:
            Processed result (e.g., probabilities, class names, masks).
        """
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        raise NotImplementedError("Adapter must implement num_classes property.")

    @property
    def num_bands(self) -> int:
        """Number of input bands expected."""
        raise NotImplementedError("Adapter must implement num_bands property.")

    @property
    def patch_size(self) -> int:
        """Input patch size (e.g. 120)."""
        raise NotImplementedError("Adapter must implement patch_size property.")

    @property
    def stride(self) -> int:
        """Sliding window stride."""
        raise NotImplementedError("Adapter must implement stride property.")

    @property
    def is_segmentation(self) -> bool:
        """True if model outputs a map (N, C, H, W), False if vector (N, C)."""
        return False

    @property
    def labels(self) -> List[str]:
        """List of class labels."""
        return []

    @property
    def color_map(self) -> Dict[str, Any]:
        """Dictionary mapping label names to RGB colors."""
        return {}
