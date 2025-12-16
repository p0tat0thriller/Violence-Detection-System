import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ViolenceDataset
from model import ViolenceModel
from tqdm import tqdm
import os


# Configuration Constants
BATCH_SIZE = 16  # Reduce to 8 if memory issues
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset paths (adjust as needed)
TRAIN_DATA_PATH = "/kaggle/input/rwf-2000/train"  # Standard Kaggle path
VAL_DATA_PATH = "/kaggle/input/rwf-2000/val"      # Standard Kaggle path

# Alternative local paths (uncomment if using local dataset)
# TRAIN_DATA_PATH = "data/train"
# VAL_DATA_PATH = "data/val"

print(f"Using device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")


def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy from model outputs and true labels.
    
    Args:
        outputs (torch.Tensor): Model predictions of shape (batch_size, num_classes)
        labels (torch.Tensor): True labels of shape (batch_size,)
        
    Returns:
        float: Accuracy as a percentage
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return (correct / total) * 100.0


def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: The ViolenceModel to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        epoch: Current epoch number
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for batch_idx, (videos, labels) in enumerate(pbar):
        # Move data to device
        videos = videos.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(videos)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Track statistics
        total_loss += loss.item()
        total_samples += labels.size(0)
        
        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, epoch):
    """
    Validate the model.
    
    Args:
        model: The ViolenceModel to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        epoch: Current epoch number
        
    Returns:
        tuple: (average_validation_loss, validation_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Progress bar
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
    
    with torch.no_grad():
        for videos, labels in pbar:
            # Move data to device
            videos = videos.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Forward pass
            outputs = model(videos)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            current_acc = (total_correct / total_samples) * 100
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = (total_correct / total_samples) * 100.0
    
    return avg_loss, accuracy


def main():
    """
    Main training function.
    """
    print("Initializing Violence Detection Training...")
    
    # Check if dataset paths exist
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Warning: Training path {TRAIN_DATA_PATH} not found!")
        print("Please adjust TRAIN_DATA_PATH in the script.")
        return
    
    if not os.path.exists(VAL_DATA_PATH):
        print(f"Warning: Validation path {VAL_DATA_PATH} not found!")
        print("Please adjust VAL_DATA_PATH in the script.")
        return
    
    # Initialize datasets
    print("Creating datasets...")
    train_dataset = ViolenceDataset(root_dir=TRAIN_DATA_PATH, sequence_length=16)
    val_dataset = ViolenceDataset(root_dir=VAL_DATA_PATH, sequence_length=16)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Adjust based on your system
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    # Initialize model
    print("Initializing model...")
    model = ViolenceModel().to(DEVICE)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Track best model
    best_val_accuracy = 0.0
    
    print("\nStarting training...")
    print("-" * 70)
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Training
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validation
        val_loss, val_accuracy = validate(model, val_loader, criterion, epoch)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.2f}%")
        
        # Checkpointing - Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # Save model state dict
            model_save_path = '../weights/best_model.pth'
            os.makedirs('../weights', exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            
            print(f"  🎉 New Best Model Saved! (Accuracy: {val_accuracy:.2f}%)")
        
        print("-" * 70)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Best model saved to: ../weights/best_model.pth")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
