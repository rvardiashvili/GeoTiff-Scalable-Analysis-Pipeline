import torch
import torch.nn as nn
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from importlib import import_module

log = logging.getLogger(__name__)

class InferenceEngine:
    """
    A generic inference engine that loads a model and its adapter from a configuration,
    then runs the inference pipeline: Preprocess -> Model -> Postprocess.
    """
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None, adapter: Any = None):
        self.config = config
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Adapter (Dependency Injection or Factory)
        if adapter:
            self.adapter = adapter
        else:
            self.adapter = self._load_adapter()
        
        # 2. Build Model
        log.info(f"Building model on device: {self.device}")
        self.model = self.adapter.build_model()
        
        # 3. Load Weights (if provided and not handled by adapter internally)
        # Note: Some adapters might load weights in build_model (e.g., from HF Hub).
        if 'model_weights_path' in self.config and self.config['model_weights_path']:
            weights_path = self.config['model_weights_path']
            log.info(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            # Handle Lightning checkpoints vs raw state dicts
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False) # strict=False to allow flexible loading
            
        self.model.to(self.device)
        self.model.eval()

    def _load_adapter(self):
        adapter_cfg = self.config.get('adapter')
        if not adapter_cfg:
            raise ValueError("Configuration missing 'adapter' section.")
            
        module_path = adapter_cfg['path']
        class_name = adapter_cfg['class']
        params = adapter_cfg.get('params', {})
        
        try:
            module = import_module(module_path)
            adapter_class = getattr(module, class_name)
            return adapter_class(params)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Could not load adapter '{class_name}' from '{module_path}': {e}")

    def preprocess(self, raw_input: Any) -> Any:
        """
        Runs only the preprocessing step (CPU bound).
        """
        return self.adapter.preprocess(raw_input)

    def predict_raw(self, model_input: Any) -> Any:
        """
        Runs inference ONLY. Returns raw model output (logits + metadata).
        Post-processing is offloaded to the writer process.
        """
        # Move tensors to device if they are tensors
        if isinstance(model_input, torch.Tensor):
            model_input_device = model_input.to(self.device)
        elif isinstance(model_input, (list, tuple)):
            model_input_device = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in model_input]
        elif isinstance(model_input, dict):
            model_input_device = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}
        else:
            model_input_device = model_input

        # 2. Inference
        with torch.no_grad():
            model_output = self.model(model_input_device)

        return model_output

    def run(self, raw_input: Any) -> Any:
        """
        Executes the full inference pipeline on the given input.
        """
        # 1. Preprocess
        model_input = self.preprocess(raw_input)
        
        # 2. Predict (Inference + Postprocess)
        return self.predict(model_input)
