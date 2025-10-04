# Project 155. Quantization-aware training
# Description:
# Quantization-Aware Training (QAT) simulates the effects of quantization during training, allowing models to learn to be robust to reduced precision (e.g., int8). Unlike post-training quantization, QAT yields higher accuracy and is better suited for edge devices, mobile, and IoT applications where low latency and small model size matter.

# Python Implementation: Quantization-Aware Training on MNIST (PyTorch 2.0+)
# Install if not already: pip install torch torchvision matplotlib
 
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qat_qconfig, prepare_qat, convert
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
 
# Prepare MNIST dataset with modern transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
])

# Create data directory
data_dir = Path('./data')
data_dir.mkdir(exist_ok=True)

train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
 
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
 
# Define modern quantizable model
class QuantizedMLP(nn.Module):
    def __init__(self):
        super(QuantizedMLP, self).__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()
 
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return self.dequant(x)
 
# Instantiate and prepare for QAT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = QuantizedMLP().to(device)
 
# Fuse layers (important for quantization)
model_fused = fuse_modules(model, [['fc1', 'relu1'], ['fc2', 'relu2']])
 
# Prepare QAT config
model_fused.qconfig = get_default_qat_qconfig('fbgemm')
model_prepared = prepare_qat(model_fused, inplace=False)
 
# Training setup
optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
 
def train_epoch(model, loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total
 
# Train with QAT
print("\n🚀 Starting Quantization-Aware Training...")
train_losses = []
train_accuracies = []

for epoch in range(3):
    loss, acc = train_epoch(model_prepared, train_loader)
    train_losses.append(loss)
    train_accuracies.append(acc)
    print(f"🧠 Epoch {epoch+1}/3: Loss: {loss:.4f}, Accuracy: {acc:.2%}")
 
# Convert to quantized model
print("\n📦 Converting to quantized model...")
model_prepared.eval()
model_prepared.cpu()
quantized_model = convert(model_prepared, inplace=False)
 
# Evaluation
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    
    return correct / total
 
# Evaluate quantized model
print("\n📊 Evaluating quantized model...")
acc = evaluate(quantized_model, test_loader)
print(f"📦 Final Quantized Model Accuracy: {acc:.2%}")

# Calculate model sizes
def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)  # MB

original_size = get_model_size(model)
quantized_size = get_model_size(quantized_model)

print(f"\n📏 Model Size Comparison:")
print(f"Original model: {original_size:.2f} MB")
print(f"Quantized model: {quantized_size:.2f} MB")
print(f"Compression ratio: {original_size/quantized_size:.2f}x")

# Create visualization
print("\n📈 Creating training visualization...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'g-', label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save plot
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)
plt.savefig(plots_dir / 'training_results.png', dpi=300, bbox_inches='tight')
print(f"Plot saved to {plots_dir / 'training_results.png'}")

print("\n✅ Training completed successfully!")
print(f"Final accuracy: {acc:.2%}")
print(f"Model compression: {original_size/quantized_size:.2f}x")