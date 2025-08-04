"""
Phase 7: Multimodal Deep Learning Model for PD Treatment Response
Author: Niket
Date: August 4, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# -------------- Model Definition --------------

class MultimodalNet(nn.Module):
    def __init__(self, input_dims):
        super(MultimodalNet, self).__init__()
        # Separate encoders for each modality
        self.speech_encoder = nn.Sequential(
            nn.Linear(input_dims['speech'], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        self.motor_encoder = nn.Sequential(
            nn.Linear(input_dims['motor'], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        self.clinical_encoder = nn.Sequential(
            nn.Linear(input_dims['clinical'], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        # Fusion & output layers
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, speech, motor, clinical):
        s = self.speech_encoder(speech)
        m = self.motor_encoder(motor)
        c = self.clinical_encoder(clinical)
        x = torch.cat([s, m, c], dim=1)
        return self.classifier(x)

# -------------- Data Preparation --------------

def prepare_data():
    df = pd.read_csv(r"C:/Users/niket/OneDrive/Desktop/filtered_data/fox_insight_multimodal_CORE_IMPUTED.csv")
    # Replace with actual column names for each set:
    speech_features   = ['PdpropBradySpeech_1', 'FIVEPDVoice']     # example
    motor_features    = ['CompDiffWalk', 'EpisodeOff', 'WalkDay']  # example
    clinical_features = ['PdpropSev_1', 'PdpropSev_2', 'PdpropSev_3', 'age']  # example
    
    X_speech   = df[speech_features].values
    X_motor    = df[motor_features].values
    X_clinical = df[clinical_features].values
    
    y = (df['CGIPD'] >= 4).astype(int).values  # Binary label: 0 (Better/Same), 1 (Worse)

    # Standardize features
    X_speech = StandardScaler().fit_transform(X_speech)
    X_motor = StandardScaler().fit_transform(X_motor)
    X_clinical = StandardScaler().fit_transform(X_clinical)

    # Convert to torch tensors
    X_speech = torch.tensor(X_speech, dtype=torch.float32)
    X_motor = torch.tensor(X_motor, dtype=torch.float32)
    X_clinical = torch.tensor(X_clinical, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X_speech, X_motor, X_clinical, y

# -------------- Training & Evaluation Loop --------------

def train():
    X_speech, X_motor, X_clinical, y = prepare_data()
    dataset = TensorDataset(X_speech, X_motor, X_clinical, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    input_dims = {'speech': X_speech.shape[1], 'motor': X_motor.shape[1], 'clinical': X_clinical.shape[1]}
    model = MultimodalNet(input_dims)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for s, m, c, labels in train_loader:
            s, m, c, labels = s.to(device), m.to(device), c.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(s, m, c)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")

    # ---- Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for s, m, c, labels in val_loader:
            s, m, c, labels = s.to(device), m.to(device), c.to(device), labels.to(device)
            outputs = model(s, m, c)
            pred = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == '__main__':
    train()
