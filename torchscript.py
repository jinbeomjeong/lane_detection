import torch
from module import LaneClassification

model = LaneClassification()
script = model.to_torchscript()

# save for use in production environment
model = torch.jit.save(script, "model.pt")