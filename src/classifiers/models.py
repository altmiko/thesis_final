# models.py
"""
Neural network architectures for IoT Intrusion Detection.

This module implements:
- SimpleMLP: Basic multilayer perceptron baseline
- CNNOnly: 1D CNN for spatial/statistical pattern learning
- LSTMOnly: LSTM/BiLSTM for temporal pattern learning
- SerialCNNLSTM: Sequential CNN-LSTM architecture
- DualPathIDS: Dual-path CNN+LSTM with attention fusion (main contribution)

All models accept input of shape (batch, num_features) and output logits.
DualPathIDS can optionally return attention weights and branch features.

Author: Research Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


class SimpleMLP(nn.Module):
    """
    Simple Multilayer Perceptron baseline.

    Architecture:
        Input (num_features) → Dense(128) → ReLU → Dropout
        → Dense(64) → ReLU → Dropout → Dense(num_classes)

    Input shape: (batch, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3
    ):
        """
        Initialize the MLP.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        layers = []
        in_dim = num_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        # x shape: (batch, num_features)
        h = self.features(x)  # (batch, hidden_dim)
        logits = self.classifier(h)  # (batch, num_classes)
        return logits


class CNNOnly(nn.Module):
    """
    1D CNN for learning spatial/statistical patterns from tabular features.

    Architecture:
        Input (batch, num_features) → Reshape to (batch, 1, num_features)
        → Conv1d(1→32, k=3) → ReLU → Conv1d(32→64, k=3) → ReLU
        → AdaptiveMaxPool → Flatten → Dense(64) → ReLU → Dropout
        → Dense(num_classes)

    The 1D convolution treats features as a 1D "signal" to capture
    local patterns between adjacent features.

    Input shape: (batch, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: Tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        fc_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize the CNN.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            conv_channels: Tuple of (first_conv_out, second_conv_out) channels
            kernel_size: Convolution kernel size
            fc_dim: Fully connected layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        # Convolutional layers with padding to preserve length
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels[0],
            kernel_size=kernel_size,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_size,
            padding=padding
        )

        # Pooling to fixed size
        self.pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fc_dim, num_classes)

        # Store output dimension for feature extraction
        self.feature_dim = fc_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features)
            return_features: If True, also return intermediate features

        Returns:
            If return_features=False: logits of shape (batch, num_classes)
            If return_features=True: (logits, features) where features is (batch, fc_dim)
        """
        # Reshape for 1D conv: (batch, num_features) → (batch, 1, num_features)
        x = x.unsqueeze(1)

        # Convolutional layers
        x = F.relu(self.conv1(x))  # (batch, 32, num_features)
        x = F.relu(self.conv2(x))  # (batch, 64, num_features)

        # Pooling
        x = self.pool(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)

        # FC layers
        features = self.fc(x)  # (batch, fc_dim)
        logits = self.classifier(features)  # (batch, num_classes)

        if return_features:
            return logits, features
        return logits


class LSTMOnly(nn.Module):
    """
    LSTM/BiLSTM for learning temporal patterns.

    For tabular data, we treat the feature vector as a sequence of length 1.
    This is designed to be flexible for future extension to true sequences.

    Architecture:
        Input (batch, num_features) → Reshape to (batch, 1, num_features)
        → BiLSTM(hidden=64) → Take last hidden state
        → Dense(64) → ReLU → Dropout → Dense(num_classes)

    Input shape: (batch, num_features) or (batch, seq_len, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        fc_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize the LSTM.

        Args:
            num_features: Number of input features (or feature dim if using sequences)
            num_classes: Number of output classes
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            fc_dim: Fully connected layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output dimension depends on bidirectional
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fc_dim, num_classes)

        # Store output dimension for feature extraction
        self.feature_dim = fc_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features) or (batch, seq_len, num_features)
            return_features: If True, also return intermediate features

        Returns:
            If return_features=False: logits of shape (batch, num_classes)
            If return_features=True: (logits, features) where features is (batch, fc_dim)
        """
        # Handle 2D input (batch, num_features) → (batch, 1, num_features)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        # LSTM forward pass
        # output: (batch, seq_len, hidden_dim * num_directions)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        output, (h_n, c_n) = self.lstm(x)

        # Use the last output (for sequences) or the only output (for single step)
        # Take output from last time step
        lstm_out = output[:, -1, :]  # (batch, hidden_dim * num_directions)

        # FC layers
        features = self.fc(lstm_out)  # (batch, fc_dim)
        logits = self.classifier(features)  # (batch, num_classes)

        if return_features:
            return logits, features
        return logits


class SerialCNNLSTM(nn.Module):
    """
    Serial CNN→LSTM architecture (baseline for comparison).

    The CNN extracts local features, which are then fed to LSTM.
    This creates an information bottleneck as noted in the research problem.

    Architecture:
        Input → Conv1d layers → LSTM → FC → Classifier

    Input shape: (batch, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: Tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        fc_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize the Serial CNN-LSTM.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            conv_channels: Tuple of conv layer output channels
            kernel_size: Convolution kernel size
            lstm_hidden: LSTM hidden dimension
            fc_dim: Fully connected layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        padding = kernel_size // 2

        # CNN layers
        self.conv1 = nn.Conv1d(1, conv_channels[0], kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size, padding=padding)

        # LSTM: treats conv output as sequence
        # Conv output shape: (batch, conv_channels[1], num_features)
        # For LSTM: (batch, seq_len=num_features, input_dim=conv_channels[1])
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = lstm_hidden * 2  # Bidirectional

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fc_dim, num_classes)
        self.feature_dim = fc_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features)
            return_features: If True, also return intermediate features

        Returns:
            logits or (logits, features) tuple
        """
        # CNN forward
        x = x.unsqueeze(1)  # (batch, 1, num_features)
        x = F.relu(self.conv1(x))  # (batch, 32, num_features)
        x = F.relu(self.conv2(x))  # (batch, 64, num_features)

        # Transpose for LSTM: (batch, num_features, 64)
        x = x.transpose(1, 2)

        # LSTM forward
        output, _ = self.lstm(x)  # (batch, num_features, lstm_hidden*2)

        # Take last output
        lstm_out = output[:, -1, :]  # (batch, lstm_hidden*2)

        # FC layers
        features = self.fc(lstm_out)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class FusionAttention(nn.Module):
    """
    Attention module for fusing CNN and LSTM branch outputs.

    Computes attention weights over the two branches to determine
    which branch to trust more for each sample.

    Methods:
        - branch_attention: Computes weights α over [h_cnn, h_lstm]
        - feature_attention: Computes weights over concatenated features

    Input: h_cnn (batch, dim), h_lstm (batch, dim)
    Output: fused (batch, dim), attention_weights (batch, 2)
    """

    def __init__(
        self,
        feature_dim: int,
        attention_type: str = 'branch',  # 'branch' or 'feature'
        hidden_dim: int = 32
    ):
        """
        Initialize the fusion attention module.

        Args:
            feature_dim: Dimension of each branch's feature vector
            attention_type: 'branch' for branch-level, 'feature' for feature-level attention
            hidden_dim: Hidden dimension for attention MLP
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.attention_type = attention_type

        if attention_type == 'branch':
            # Branch-level attention: compute 2 weights
            # Input: concatenated [h_cnn, h_lstm] of size 2*feature_dim
            self.attention_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2)  # 2 branches
            )
        else:  # feature-level attention
            # Feature-level: compute weight for each feature in concatenated vector
            self.attention_mlp = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, feature_dim * 2)
            )

    def forward(
        self,
        h_cnn: torch.Tensor,
        h_lstm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention-weighted fusion of branch outputs.

        Args:
            h_cnn: CNN branch features of shape (batch, feature_dim)
            h_lstm: LSTM branch features of shape (batch, feature_dim)

        Returns:
            Tuple of:
            - fused: Fused feature vector of shape (batch, feature_dim) for branch attention
                     or (batch, feature_dim*2) for feature attention
            - attention_weights: Shape (batch, 2) for branch or (batch, feature_dim*2) for feature
        """
        # Concatenate branch outputs
        combined = torch.cat([h_cnn, h_lstm], dim=1)  # (batch, feature_dim*2)

        if self.attention_type == 'branch':
            # Compute branch attention weights
            attention_logits = self.attention_mlp(combined)  # (batch, 2)
            attention_weights = F.softmax(attention_logits, dim=1)  # (batch, 2)

            # Weighted sum of branches
            # Stack branches: (batch, 2, feature_dim)
            stacked = torch.stack([h_cnn, h_lstm], dim=1)
            # Expand weights: (batch, 2, 1)
            weights_expanded = attention_weights.unsqueeze(-1)
            # Weighted sum: (batch, feature_dim)
            fused = (stacked * weights_expanded).sum(dim=1)

            return fused, attention_weights

        else:  # feature-level attention
            # Compute per-feature attention weights
            attention_logits = self.attention_mlp(combined)  # (batch, feature_dim*2)
            attention_weights = torch.sigmoid(attention_logits)  # (batch, feature_dim*2)

            # Element-wise weighted features
            fused = combined * attention_weights

            return fused, attention_weights


class CNNBranch(nn.Module):
    """
    CNN branch for the dual-path architecture.

    Processes input through 1D convolutions and returns features.
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: Tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_channels[1], output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, num_features)

        Returns:
            Features of shape (batch, output_dim)
        """
        x = x.unsqueeze(1)  # (batch, 1, num_features)
        x = self.conv_layers(x)  # (batch, 64, 1)
        x = x.squeeze(-1)  # (batch, 64)
        x = self.fc(x)  # (batch, output_dim)
        return x


class LSTMBranch(nn.Module):
    """
    LSTM branch for the dual-path architecture.

    Processes input through BiLSTM and returns features.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = hidden_dim * 2

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, num_features) or (batch, seq_len, num_features)

        Returns:
            Features of shape (batch, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, num_features)

        output, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        lstm_out = output[:, -1, :]  # (batch, hidden*2)
        features = self.fc(lstm_out)  # (batch, output_dim)
        return features


class DualPathIDS(nn.Module):
    """
    Dual-Path CNN+LSTM with Attention Fusion for Intrusion Detection.

    This is the main proposed architecture that addresses the information
    bottleneck problem in serial CNN→LSTM architectures.

    Architecture:
        Input ──┬── CNN Branch ── h_cnn ──┐
                │                          ├── Attention Fusion ── Classifier
                └── LSTM Branch ─ h_lstm ──┘

    The attention mechanism learns to weight the branches based on input,
    allowing the model to trust more robust patterns when under attack.

    Input shape: (batch, num_features)
    Output: logits, and optionally (attention_weights, (h_cnn, h_lstm))
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        branch_dim: int = 64,
        attention_type: str = 'branch',  # 'branch' or 'feature'
        dropout: float = 0.3
    ):
        """
        Initialize the Dual-Path IDS model.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            branch_dim: Output dimension for each branch
            attention_type: Type of attention fusion ('branch' or 'feature')
            dropout: Dropout probability
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.attention_type = attention_type

        # Dual branches
        self.cnn_branch = CNNBranch(
            num_features=num_features,
            output_dim=branch_dim,
            dropout=dropout
        )

        self.lstm_branch = LSTMBranch(
            num_features=num_features,
            output_dim=branch_dim,
            dropout=dropout
        )

        # Attention fusion
        self.attention = FusionAttention(
            feature_dim=branch_dim,
            attention_type=attention_type
        )

        # Classifier
        # Output dim depends on attention type
        if attention_type == 'branch':
            classifier_input_dim = branch_dim
        else:
            classifier_input_dim = branch_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, num_classes)
        )

        self.branch_dim = branch_dim

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ]:
        """
        Forward pass through the dual-path architecture.

        Args:
            x: Input tensor of shape (batch, num_features)
            return_attention: If True, also return attention weights
            return_features: If True, also return branch features (h_cnn, h_lstm)

        Returns:
            - If both False: logits of shape (batch, num_classes)
            - If return_attention=True: (logits, attention_weights)
            - If both True: (logits, attention_weights, (h_cnn, h_lstm))
        """
        # Process through both branches
        h_cnn = self.cnn_branch(x)  # (batch, branch_dim)
        h_lstm = self.lstm_branch(x)  # (batch, branch_dim)

        # Attention fusion
        fused, attention_weights = self.attention(h_cnn, h_lstm)

        # Classification
        logits = self.classifier(fused)  # (batch, num_classes)

        # Return based on flags
        if return_attention and return_features:
            return logits, attention_weights, (h_cnn, h_lstm)
        elif return_attention:
            return logits, attention_weights
        else:
            return logits


class AttentionSerialCNNLSTM(nn.Module):
    """
    Serial CNN→LSTM with simple attention (no adversarial training).

    Baseline for comparing with the dual-path architecture.
    Uses attention over LSTM hidden states rather than branch-level attention.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: Tuple[int, int] = (32, 64),
        lstm_hidden: int = 64,
        fc_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        # CNN layers
        self.conv1 = nn.Conv1d(1, conv_channels[0], 3, padding=1)
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], 3, padding=1)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = lstm_hidden * 2

        # Simple attention over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 1),
            nn.Softmax(dim=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fc_dim, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with optional attention weights return."""

        # CNN
        x = x.unsqueeze(1)  # (batch, 1, num_features)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.transpose(1, 2)  # (batch, num_features, channels)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_out_dim)

        # Attention
        att_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        context = (lstm_out * att_weights).sum(dim=1)  # (batch, lstm_out_dim)

        # Classification
        features = self.fc(context)
        logits = self.classifier(features)

        if return_attention:
            return logits, att_weights.squeeze(-1)
        return logits


def get_model(
    model_type: str,
    num_features: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_type: One of 'mlp', 'cnn', 'lstm', 'serial', 'dualpath', 'attn_serial'
        num_features: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        'mlp': SimpleMLP,
        'cnn': CNNOnly,
        'lstm': LSTMOnly,
        'serial': SerialCNNLSTM,
        'dualpath': DualPathIDS,
        'attn_serial': AttentionSerialCNNLSTM
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(models.keys())}")

    return models[model_type](num_features, num_classes, **kwargs)


if __name__ == '__main__':
    # Quick test of all models
    batch_size = 16
    num_features = 45
    num_classes = 6

    x = torch.randn(batch_size, num_features)

    print("Testing all models...")

    for name in ['mlp', 'cnn', 'lstm', 'serial', 'dualpath', 'attn_serial']:
        model = get_model(name, num_features, num_classes)
        output = model(x)
        print(f"{name:15} output shape: {output.shape}")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:15} parameters: {n_params:,}")

    # Test DualPathIDS with attention output
    print("\nTesting DualPathIDS with full outputs...")
    model = DualPathIDS(num_features, num_classes)
    logits, attn, (h_cnn, h_lstm) = model(x, return_attention=True, return_features=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Attention shape: {attn.shape}")
    print(f"h_cnn shape: {h_cnn.shape}")
    print(f"h_lstm shape: {h_lstm.shape}")
    print(f"Attention weights sample: {attn[0]}")
