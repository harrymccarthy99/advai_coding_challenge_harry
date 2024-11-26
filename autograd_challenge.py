import numpy as np

# Custom Imports
from src.utils import mse_loss
from src.custom_grad import Tensor
from src.simple_nn import NN, Optimizer


# Initialise model
model = NN(nin = 5, nouts = [10, 8, 5, 1])
print("---------")
print("Model Layers;")
print(model)


inputs = Tensor([[0.5, -0.2, 0.1, 0.7, -0.3]])
target = Tensor([[1.0]])
EPOCHS = 1
optimizer = Optimizer(params=model.parameters(), lr=0.01)

# Train for one epoch
y_pred = model(inputs)  # Forward pass
train_loss = mse_loss(y_pred, target)  # Calculate loss

optimizer.zero_grad()  # Clear gradients
train_loss.backward()  # Backward pass

optimizer.step()       # Update parameters

# Print weights of the first layer
first_layer_weights = model.layers[0].W.data
print("---------")
print("\nWeights of the first layer after one epoch;")
print(first_layer_weights.reshape(10, 5))
print(f"Model prediction after 1 epoch; {y_pred.data}")


# Training model for longer now

for epoch in range(100):
    y_pred = model(inputs)  # Forward pass
    train_loss = mse_loss(y_pred, target)  # Calculate loss

    optimizer.zero_grad()  # Clear gradients
    train_loss.backward()  # Backward pass
    optimizer.step()       # Update parameters


# Print weights of the first layer after training
first_layer_weights = model.layers[0].W.data
print("---------")
print("\nWeights of the first layer after 100 epochs;")
print(first_layer_weights.reshape(10, 5))
# Final prediction
print(f"Model prediction after 100 epochs: {y_pred.data}")
print("---------")
