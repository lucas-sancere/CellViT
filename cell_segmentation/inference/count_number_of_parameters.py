import torch

# Load the checkpoint (change path)


pth = False
tar = True 

### IF file is .pth
if pth:
    ckpt = torch.load("/data/lsancere/Data_General/Predictions/NucSeg_ValSplit_CellViTFormat/val_evaluate_cellvit256/checkpoints/model_best.pth", map_location='cpu')
    
    # Extract the model weights
    state_dict = ckpt["model_state_dict"]

    # Count only tensor values
    total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))

    print(f"Total parameters: {total_params:,}")

if tar:
    ckpt = torch.load("/data/lsancere/Data_General/Weights/bestcheckpoints-histo-miner/scchovernet_bestweights.tar", map_location='cpu')

    # Extract model weights
    state_dict = ckpt["model_state_dict"]

    # Count parameters
    total_params = sum(v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor))
    print(f"Total parameters: {total_params:,}")