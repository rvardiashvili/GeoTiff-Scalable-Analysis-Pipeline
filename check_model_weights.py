
import torch
from huggingface_hub import hf_hub_download
import safetensors.torch
import sys

def check_weights(model_id):
    print(f"Checking {model_id}...")
    try:
        path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = safetensors.torch.load_file(path, device='cpu')
    except:
        try:
            path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
            state_dict = torch.load(path, map_location='cpu')
        except Exception as e:
            print(f"Could not download/load {model_id}: {e}")
            return

    # Find first conv layer
    for k, v in state_dict.items():
        # ResNet usually 'conv1.weight' or 'stem.0.weight'
        # ConvNeXt usually 'stem.0.weight'
        if 'conv1.weight' in k or 'stem.0.weight' in k or 'patch_embed.proj.weight' in k:
            print(f"  Layer: {k} | Shape: {v.shape}")
            if v.shape[1] == 12:
                print("  -> EXPECTS 12 BANDS")
            elif v.shape[1] == 10:
                print("  -> EXPECTS 10 BANDS")
            else:
                print(f"  -> EXPECTS {v.shape[1]} BANDS")
            return

if __name__ == "__main__":
    check_weights("BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0")
    check_weights("BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0")
