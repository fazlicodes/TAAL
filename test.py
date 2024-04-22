import torch
from models import EmbModel


# Load the checkpoint
checkpoint = torch.load(f'/l/users/sanoojan.baliah/Felix/svl_adapter/dino_rn50/eurosat/eurosat-net.pt')

# Extract the state dictionary
state_dict = checkpoint

# Create the model
base_encoder = eval('resnet50')
model = EmbModel(base_encoder, model_args).to(args['device'])

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Optionally, if you want to ensure that the loading process is successful
# you can print a message indicating so
print("Model loaded successfully.")
