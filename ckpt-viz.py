import torch

ckpt = torch.load("aptos-epoch04-accuracy-0.7832-f1-0.1757.ckpt", map_location="cpu")

# sometimes checkpoint may be nested inside 'state_dict'
if "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

print("Number of keys in checkpoint:", len(state_dict))
print("Some example keys:")
print(list(state_dict.keys())[:20])
