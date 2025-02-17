# scripts/train_alias.py
import torch
import torch.optim as optim
import torch.nn as nn
from models.appearance_flow import AppearanceFlow
from utils.dataset import VITONDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm  # Import tqdm for progress bar

train_data = VITONDataset("datasets/train_pairs.txt")
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2)  # Changed batch size...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Modified for CPU
model = AppearanceFlow().to(device)  # CPU
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

num_epochs = 5  # Changed epochs from 10 to 1.

for epoch in range(num_epochs):
    epoch_loss = 0.0  # Track total loss for the epoch
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)  # Create progress bar

    for images, targets in progress_bar:
        images, masks = images.to(device), masks.squeeze(1).squeeze(-1).long().to(device)  # Changed
        masks = torch.clamp(masks, min=0, max=12)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")  # Update progress bar with loss

    print(f"Epoch {epoch+1}, Avg Loss: {epoch_loss / len(train_loader):.4f}")  # Print epoch loss

#torch.save(model.state_dict(), "checkpoints/alias_model.pth")
if torch.cuda.is_available():  # Commented to run on CPU
    torch.save(model.state_dict(), "checkpoints/alias_model.pth")
    print("Model saved.")
else:
    print("Skipping model save (testing on CPU).")

