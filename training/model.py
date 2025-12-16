import torch
import torch.nn as nn
import torchvision.models


class ViolenceModel(nn.Module):
    def __init__(self):
        """
        Initialize the ViolenceModel with ResNet50 feature extractor and LSTM classifier.
        
        Architecture:
        - ResNet50 (frozen) as feature extractor: (batch*seq, 3, 224, 224) -> (batch*seq, 2048)
        - LSTM: (batch, seq, 2048) -> (batch, seq, 512)
        - Linear classifier: (batch, 512) -> (batch, 2)
        """
        super(ViolenceModel, self).__init__()
        
        # Load pre-trained ResNet50 model
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        
        # Remove the last fully connected layer to get feature extractor
        # ResNet50 output before FC layer is 2048 features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet50 parameters to speed up training
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # LSTM layer for temporal modeling
        self.lstm = nn.LSTM(
            input_size=2048,      # ResNet50 feature size
            hidden_size=512,      # LSTM hidden size
            num_layers=2,         # Number of LSTM layers
            batch_first=True,     # Input shape: (batch, seq, features)
            dropout=0.3           # Dropout for regularization
        )
        
        # Final classifier layer
        self.classifier = nn.Linear(512, 2)  # 512 -> 2 (Violence/NonViolence)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass of the ViolenceModel.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 2)
        """
        # Input shape: (batch_size, sequence_length, 3, 224, 224)
        batch_size, sequence_length, channels, height, width = x.size()
        
        # Step 1: Reshape for CNN processing
        # Combine batch and sequence dimensions to treat all frames independently
        x = x.view(batch_size * sequence_length, channels, height, width)
        # Shape: (batch_size * sequence_length, 3, 224, 224)
        
        # Step 2: Extract features using ResNet50
        with torch.no_grad():  # Since ResNet is frozen, no gradients needed
            features = self.feature_extractor(x)
        # Shape: (batch_size * sequence_length, 2048, 1, 1)
        
        # Flatten the spatial dimensions
        features = features.view(batch_size * sequence_length, -1)
        # Shape: (batch_size * sequence_length, 2048)
        
        # Step 3: Reshape back to sequence format for LSTM
        features = features.view(batch_size, sequence_length, -1)
        # Shape: (batch_size, sequence_length, 2048)
        
        # Step 4: Pass through LSTM
        lstm_out, (hidden, cell) = self.lstm(features)
        # lstm_out shape: (batch_size, sequence_length, 512)
        
        # Step 5: Extract the last time step output
        # Use the hidden state of the last frame
        last_output = lstm_out[:, -1, :]  # Get last time step
        # Shape: (batch_size, 512)
        
        # Apply dropout for regularization
        last_output = self.dropout(last_output)
        
        # Step 6: Pass through classifier
        logits = self.classifier(last_output)
        # Shape: (batch_size, 2)
        
        return logits
    
    def get_num_params(self):
        """
        Get the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_feature_extractor(self):
        """
        Freeze the ResNet50 feature extractor parameters.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_feature_extractor(self):
        """
        Unfreeze the ResNet50 feature extractor parameters for fine-tuning.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def set_feature_extractor_trainable(self, trainable=True):
        """
        Set the trainability of the feature extractor.
        
        Args:
            trainable (bool): Whether to make feature extractor trainable
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = trainable


# Test function to verify model architecture
def test_model():
    """
    Test function to verify the model works with expected input shapes.
    """
    # Create model
    model = ViolenceModel()
    
    # Create dummy input: (batch_size=2, sequence_length=16, channels=3, height=224, width=224)
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model created successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of trainable parameters: {model.get_num_params():,}")
    
    return model


if __name__ == "__main__":
    test_model()
