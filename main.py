from MTLFormer import MTLFormer
from train import train_model
from eval import evaluate_model
from torch import optim


# Model hyperparameters
embed_size = 128
heads = 8
dropout = 0.3
num_classes = 14  # For next activity prediction

# Instantiate model
model = MTLFormer(embed_size, heads, dropout, num_classes)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.002)

# Define loss weights (tuning these may help balance task performance)
weights = [0.6, 0.2, 0.2]  # Can be tuned

# Dataloaders (to be defined based on your dataset)
# train_loader = DataLoader(...)
# val_loader = DataLoader(...)

# Train the model
train_model(model, train_loader, optimizer, weights, num_epochs=100)

# Evaluate the model
evaluate_model(model, val_loader)
