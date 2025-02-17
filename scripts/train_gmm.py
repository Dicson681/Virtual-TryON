import torch
import torch.optim as optim
import torch.nn as nn
from models.geometric_matching import GeometricMatchingModel
from utils.dataset import VITONDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # Import tqdm for progress bar

print("training gmm")
train_data = VITONDataset("datasets/train_pairs.txt")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Modified for CPU
model = GeometricMatchingModel().to(device)  # Move model to device
print(f"Using {device} for training.")  # Print device info

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

num_epochs = 5  # Changed from 1 back to 10 for training

for epoch in range(num_epochs):
    epoch_loss = 0.0  # Track total loss for the epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)  # Create progress bar

    for images, targets in progress_bar:
        images, masks = images.to(device), masks.squeeze(1).squeeze(-1).long().to(device)  # Changed
        masks = torch.clamp(masks, min=0, max=12)  # Ensure masks are in valid range

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")  # Update progress bar with loss

    print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(train_loader):.4f}")  # Print epoch loss

# Save model only if running on CUDA
if torch.cuda.is_available():
    torch.save(model.state_dict(), "checkpoints/gmm_model.pth")
    print("Model saved.")
else:
    print("Skipping model save (testing on CPU).")
