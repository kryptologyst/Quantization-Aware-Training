#!/usr/bin/env python3
"""
Demo script for Quantization-Aware Training
Quick demonstration of the modern QAT implementation
"""

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules, get_default_qat_qconfig, prepare_qat, convert
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_demo_model():
    """Create a simple quantized model for demonstration"""
    class DemoQuantizedMLP(nn.Module):
        def __init__(self):
            super(DemoQuantizedMLP, self).__init__()
            self.quant = QuantStub()
            self.fc1 = nn.Linear(28*28, 128)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(128, 64)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(64, 10)
            self.dequant = DeQuantStub()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.quant(x)
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return self.dequant(x)
    
    return DemoQuantizedMLP()

def demo_qat():
    """Demonstrate Quantization-Aware Training"""
    print("🧠 Quantization-Aware Training Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("\n📊 Preparing MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\n🏗️ Creating quantized model...")
    model = create_demo_model().to(device)
    
    # Fuse layers for quantization
    model_fused = fuse_modules(model, [['fc1', 'relu1'], ['fc2', 'relu2']])
    
    # Set quantization configuration
    model_fused.qconfig = get_default_qat_qconfig('fbgemm')
    
    # Prepare for QAT
    model_prepared = prepare_qat(model_fused, inplace=False)
    
    print("Model prepared for quantization-aware training!")
    
    # Training setup
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("\n🚀 Starting training...")
    model_prepared.train()
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(3):  # Quick demo with 3 epochs
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit batches for demo
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = epoch_loss / min(100, len(train_loader))
        accuracy = correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        print(f"Epoch {epoch+1}/3: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
    
    # Convert to quantized model
    print("\n📦 Converting to quantized model...")
    model_prepared.eval()
    model_prepared.cpu()
    quantized_model = convert(model_prepared, inplace=False)
    
    # Evaluate quantized model
    print("\n📊 Evaluating quantized model...")
    quantized_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = quantized_model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    final_accuracy = correct / total
    print(f"Final quantized model accuracy: {final_accuracy:.2%}")
    
    # Calculate model size
    def get_model_size(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)  # MB
    
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"\n📏 Model size comparison:")
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
    plt.savefig(plots_dir / 'demo_training.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plots_dir / 'demo_training.png'}")
    
    # Summary
    print("\n✅ Demo completed successfully!")
    print(f"Final accuracy: {final_accuracy:.2%}")
    print(f"Model compression: {original_size/quantized_size:.2f}x")
    print("\n🎯 Key takeaways:")
    print("- QAT maintains high accuracy with quantized models")
    print("- Significant model size reduction achieved")
    print("- Ready for deployment on edge devices")
    
    return {
        'final_accuracy': final_accuracy,
        'original_size_mb': original_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': original_size/quantized_size,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies
    }

if __name__ == "__main__":
    results = demo_qat()
