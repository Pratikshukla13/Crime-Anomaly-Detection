import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report
import numpy as np
import os
import pickle
from tqdm import tqdm
from classifier import CrimeClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Custom Dataset
class C3DFeaturesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ✅ Load your saved C3D features and labels
with open('../features/features.pkl', 'rb') as f:
    data_dict = pickle.load(f)

train_features, train_labels = data_dict['train_features'], data_dict['train_labels']
test_features, test_labels = data_dict['test_features'], data_dict['test_labels']

# ✅ Dataset & balancing
train_dataset = C3DFeaturesDataset(train_features, train_labels)
test_dataset = C3DFeaturesDataset(test_features, test_labels)

# Handle class imbalance
class_counts = np.bincount(train_labels)
weights = 1. / class_counts
sample_weights = [weights[label] for label in train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# ✅ Model, Loss, Optimizer
model = CrimeClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training Loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # ✅ Test accuracy
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Test Accuracy: {correct / total:.2f}")

# ✅ Report & Save
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['abuse', 'assault', 'arson', 'arrest']))

os.makedirs('../pretrained_models', exist_ok=True)
torch.save(model.state_dict(), '../pretrained_models/crime_classifier.pt')
print("\n✅ Classifier saved at: ../pretrained_models/crime_classifier.pt")
