#!/usr/bin/env python3
"""
Modern Quantization-Aware Training Implementation
Project 155: Advanced QAT with PyTorch 2.0+ APIs
"""

import os
import sys
import yaml
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    QuantStub, DeQuantStub, fuse_modules,
    get_default_qat_qconfig, prepare_qat, convert
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qat_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MockDatabase:
    """Mock database for storing training results and model metadata"""
    
    def __init__(self, db_path: str = "qat_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                config TEXT,
                model_type TEXT,
                quantization_backend TEXT,
                final_accuracy REAL,
                model_size_mb REAL,
                training_time_seconds REAL
            )
        ''')
        
        # Create training_logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                epoch INTEGER,
                train_loss REAL,
                train_accuracy REAL,
                val_loss REAL,
                val_accuracy REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_experiment(self, config: Dict, results: Dict) -> int:
        """Save experiment results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO experiments 
            (timestamp, config, model_type, quantization_backend, final_accuracy, model_size_mb, training_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(config),
            config.get('model', {}).get('type', 'QuantizedMLP'),
            config.get('quantization', {}).get('backend', 'fbgemm'),
            results.get('final_accuracy', 0.0),
            results.get('model_size_mb', 0.0),
            results.get('training_time_seconds', 0.0)
        ))
        
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return experiment_id
    
    def save_training_log(self, experiment_id: int, epoch: int, metrics: Dict):
        """Save training metrics for an epoch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO training_logs 
            (experiment_id, epoch, train_loss, train_accuracy, val_loss, val_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id,
            epoch,
            metrics.get('train_loss', 0.0),
            metrics.get('train_accuracy', 0.0),
            metrics.get('val_loss', 0.0),
            metrics.get('val_accuracy', 0.0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment_history(self) -> pd.DataFrame:
        """Get all experiment history"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT e.*, 
                   COUNT(tl.id) as total_epochs,
                   AVG(tl.train_accuracy) as avg_train_accuracy,
                   AVG(tl.val_accuracy) as avg_val_accuracy
            FROM experiments e
            LEFT JOIN training_logs tl ON e.id = tl.experiment_id
            GROUP BY e.id
            ORDER BY e.timestamp DESC
        ''', conn)
        conn.close()
        return df


class ModernQuantizedMLP(nn.Module):
    """Modern quantized MLP with improved architecture"""
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [256, 128], 
                 output_size: int = 10, dropout_rate: float = 0.2):
        super(ModernQuantizedMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.quant(x)
        
        for i in range(0, len(self.layers) - 1, 4):  # Process in groups of 4
            x = self.layers[i](x)      # Linear
            x = self.layers[i+1](x)    # BatchNorm
            x = self.layers[i+2](x)    # ReLU
            x = self.layers[i+3](x)    # Dropout
        
        # Output layer
        x = self.layers[-1](x)
        return self.dequant(x)
    
    def fuse_model(self):
        """Fuse layers for quantization"""
        # For simplicity, we'll fuse linear + relu pairs
        modules_to_fuse = []
        for i in range(0, len(self.layers) - 1, 4):
            modules_to_fuse.append([f'layers.{i}', f'layers.{i+2}'])
        
        if modules_to_fuse:
            fuse_modules(self, modules_to_fuse, inplace=True)


class QATTrainer:
    """Quantization-Aware Training trainer with modern features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.db = MockDatabase()
        
        # Setup logging
        if config['logging']['use_tensorboard']:
            self.writer = SummaryWriter(config['logging']['log_dir'])
        else:
            self.writer = None
        
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare MNIST dataset with modern transforms"""
        transform_list = [transforms.ToTensor()]
        
        if self.config['data']['normalize']:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        if self.config['data']['augment']:
            transform_list.insert(0, transforms.RandomRotation(10))
            transform_list.insert(1, transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        
        transform = transforms.Compose(transform_list)
        
        train_dataset = datasets.MNIST(
            self.config['data']['data_dir'],
            train=True,
            download=self.config['data']['download'],
            transform=transform
        )
        
        test_dataset = datasets.MNIST(
            self.config['data']['data_dir'],
            train=False,
            download=self.config['data']['download'],
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['test_batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def create_model(self) -> nn.Module:
        """Create and prepare quantized model"""
        model = ModernQuantizedMLP(
            input_size=self.config['model']['input_size'],
            hidden_sizes=self.config['model']['hidden_sizes'],
            output_size=self.config['model']['output_size'],
            dropout_rate=self.config['model']['dropout_rate']
        ).to(self.device)
        
        # Fuse layers for quantization
        model.fuse_model()
        
        # Set quantization configuration
        backend = self.config['quantization']['backend']
        model.qconfig = get_default_qat_qconfig(backend)
        
        # Prepare for QAT
        prepare_qat(model, inplace=True)
        
        return model
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Dict:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % self.config['logging']['log_interval'] == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
                
                if self.writer:
                    self.writer.add_scalar('Train/Loss', loss.item(), 
                                         len(train_loader) * self.current_epoch + batch_idx)
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total
        }
    
    def evaluate(self, model: nn.Module, test_loader: DataLoader, 
                criterion: nn.Module) -> Dict:
        """Evaluate model on test set"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return {
            'val_loss': total_loss / len(test_loader),
            'val_accuracy': correct / total
        }
    
    def train(self) -> Dict:
        """Main training loop"""
        logger.info("Starting Quantization-Aware Training")
        
        # Prepare data
        train_loader, test_loader = self.prepare_data()
        
        # Create model
        model = self.create_model()
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config['training']['scheduler_step_size'],
            gamma=self.config['training']['scheduler_gamma']
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        start_time = datetime.now()
        experiment_id = self.db.save_experiment(self.config, {})
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # Evaluate
            val_metrics = self.evaluate(model, test_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}: "
                       f"Train Loss: {metrics['train_loss']:.4f}, "
                       f"Train Acc: {metrics['train_accuracy']:.4f}, "
                       f"Val Loss: {metrics['val_loss']:.4f}, "
                       f"Val Acc: {metrics['val_accuracy']:.4f}")
            
            # Save to database
            self.db.save_training_log(experiment_id, epoch, metrics)
            
            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Epoch/Train_Loss', metrics['train_loss'], epoch)
                self.writer.add_scalar('Epoch/Train_Accuracy', metrics['train_accuracy'], epoch)
                self.writer.add_scalar('Epoch/Val_Loss', metrics['val_loss'], epoch)
                self.writer.add_scalar('Epoch/Val_Accuracy', metrics['val_accuracy'], epoch)
                self.writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Convert to quantized model
        logger.info("Converting to quantized model...")
        model.eval()
        model.cpu()
        quantized_model = convert(model, inplace=False)
        
        # Final evaluation
        final_metrics = self.evaluate(quantized_model, test_loader, criterion)
        
        # Calculate model size
        model_size_mb = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / (1024 * 1024)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'final_accuracy': final_metrics['val_accuracy'],
            'model_size_mb': model_size_mb,
            'training_time_seconds': training_time,
            'quantized_model': quantized_model
        }
        
        # Update experiment in database
        self.db.save_experiment(self.config, results)
        
        logger.info(f"Training completed! Final accuracy: {final_metrics['val_accuracy']:.4f}")
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        logger.info(f"Training time: {training_time:.2f} seconds")
        
        return results


def create_visualizations(results: Dict, config: Dict):
    """Create visualizations for training results"""
    # Get experiment history
    db = MockDatabase()
    history_df = db.get_experiment_history()
    
    if len(history_df) == 0:
        logger.warning("No experiment history found for visualization")
        return
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Plot 1: Training Progress
    plt.figure(figsize=(12, 8))
    
    # Get training logs for the latest experiment
    conn = sqlite3.connect("qat_results.db")
    logs_df = pd.read_sql_query('''
        SELECT * FROM training_logs 
        WHERE experiment_id = (SELECT MAX(id) FROM experiments)
        ORDER BY epoch
    ''', conn)
    conn.close()
    
    if not logs_df.empty:
        plt.subplot(2, 2, 1)
        plt.plot(logs_df['epoch'], logs_df['train_loss'], label='Train Loss')
        plt.plot(logs_df['epoch'], logs_df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(logs_df['epoch'], logs_df['train_accuracy'], label='Train Accuracy')
        plt.plot(logs_df['epoch'], logs_df['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
    
    # Plot 2: Model Size Comparison
    plt.subplot(2, 2, 3)
    plt.bar(['Quantized Model'], [results['model_size_mb']])
    plt.ylabel('Model Size (MB)')
    plt.title('Quantized Model Size')
    plt.grid(True, axis='y')
    
    # Plot 3: Accuracy Comparison
    plt.subplot(2, 2, 4)
    plt.bar(['Final Accuracy'], [results['final_accuracy']])
    plt.ylabel('Accuracy')
    plt.title('Final Model Accuracy')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {plots_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Quantization-Aware Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations after training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create necessary directories
    Path(config['data']['data_dir']).mkdir(exist_ok=True)
    Path(config['logging']['log_dir']).mkdir(exist_ok=True)
    Path(config['logging']['model_save_path']).mkdir(exist_ok=True)
    
    # Initialize trainer
    trainer = QATTrainer(config)
    
    # Train model
    results = trainer.train()
    
    # Create visualizations if requested
    if args.visualize:
        create_visualizations(results, config)
    
    # Save quantized model
    if config['logging']['save_model']:
        model_path = Path(config['logging']['model_save_path']) / 'quantized_model.pth'
        torch.save(results['quantized_model'].state_dict(), model_path)
        logger.info(f"Quantized model saved to {model_path}")


if __name__ == "__main__":
    main()
