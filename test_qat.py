#!/usr/bin/env python3
"""
Test suite for Quantization-Aware Training project
Comprehensive tests for all components
"""

import unittest
import tempfile
import shutil
import os
import sqlite3
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qat_trainer import MockDatabase, ModernQuantizedMLP, QATTrainer


class TestMockDatabase(unittest.TestCase):
    """Test cases for MockDatabase class"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_qat.db")
        self.db = MockDatabase(self.db_path)
    
    def tearDown(self):
        """Clean up test database"""
        shutil.rmtree(self.temp_dir)
    
    def test_database_initialization(self):
        """Test database initialization"""
        self.assertTrue(os.path.exists(self.db_path))
        
        # Check if tables exist
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('experiments', tables)
        self.assertIn('training_logs', tables)
        
        conn.close()
    
    def test_save_experiment(self):
        """Test saving experiment data"""
        config = {
            'model': {'type': 'test_model'},
            'quantization': {'backend': 'fbgemm'}
        }
        results = {
            'final_accuracy': 0.95,
            'model_size_mb': 1.5,
            'training_time_seconds': 120.0
        }
        
        experiment_id = self.db.save_experiment(config, results)
        self.assertIsInstance(experiment_id, int)
        self.assertGreater(experiment_id, 0)
    
    def test_save_training_log(self):
        """Test saving training log data"""
        # First save an experiment
        config = {'model': {'type': 'test'}}
        experiment_id = self.db.save_experiment(config, {})
        
        # Save training log
        metrics = {
            'train_loss': 0.5,
            'train_accuracy': 0.8,
            'val_loss': 0.6,
            'val_accuracy': 0.75
        }
        
        self.db.save_training_log(experiment_id, 0, metrics)
        
        # Verify data was saved
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_logs WHERE experiment_id = ?", (experiment_id,))
        result = cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result[2], 0)  # epoch
        self.assertEqual(result[3], 0.5)  # train_loss
        
        conn.close()
    
    def test_get_experiment_history(self):
        """Test retrieving experiment history"""
        # Save multiple experiments
        configs = [
            {'model': {'type': 'model1'}, 'quantization': {'backend': 'fbgemm'}},
            {'model': {'type': 'model2'}, 'quantization': {'backend': 'qnnpack'}}
        ]
        
        results = [
            {'final_accuracy': 0.9, 'model_size_mb': 1.0, 'training_time_seconds': 100},
            {'final_accuracy': 0.95, 'model_size_mb': 1.2, 'training_time_seconds': 110}
        ]
        
        for config, result in zip(configs, results):
            self.db.save_experiment(config, result)
        
        # Get history
        history_df = self.db.get_experiment_history()
        
        self.assertEqual(len(history_df), 2)
        self.assertIn('final_accuracy', history_df.columns)
        self.assertIn('model_size_mb', history_df.columns)


class TestModernQuantizedMLP(unittest.TestCase):
    """Test cases for ModernQuantizedMLP class"""
    
    def setUp(self):
        """Set up test model"""
        self.model = ModernQuantizedMLP(
            input_size=784,
            hidden_sizes=[128, 64],
            output_size=10,
            dropout_rate=0.2
        )
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_size, 784)
        self.assertEqual(self.model.hidden_sizes, [128, 64])
        self.assertEqual(self.model.output_size, 10)
        
        # Check if quantization stubs exist
        self.assertIsInstance(self.model.quant, torch.ao.quantization.QuantStub)
        self.assertIsInstance(self.model.dequant, torch.ao.quantization.DeQuantStub)
    
    def test_forward_pass(self):
        """Test forward pass"""
        batch_size = 32
        input_tensor = torch.randn(batch_size, 28, 28)
        
        with torch.no_grad():
            output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (batch_size, 10))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_fuse_model(self):
        """Test model fusion"""
        # This should not raise an exception
        self.model.fuse_model()
        
        # Check if model still works after fusion
        input_tensor = torch.randn(1, 28, 28)
        with torch.no_grad():
            output = self.model(input_tensor)
        
        self.assertEqual(output.shape, (1, 10))


class TestQATTrainer(unittest.TestCase):
    """Test cases for QATTrainer class"""
    
    def setUp(self):
        """Set up test trainer"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test config
        self.config = {
            'model': {
                'input_size': 784,
                'hidden_sizes': [128, 64],
                'output_size': 10,
                'dropout_rate': 0.2
            },
            'training': {
                'batch_size': 32,
                'test_batch_size': 100,
                'epochs': 1,  # Short training for testing
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'scheduler_step_size': 5,
                'scheduler_gamma': 0.5
            },
            'quantization': {
                'backend': 'fbgemm',
                'observer_type': 'histogram',
                'quantize_activations': True,
                'quantize_weights': True,
                'calibration_samples': 100
            },
            'data': {
                'dataset': 'MNIST',
                'data_dir': os.path.join(self.temp_dir, 'data'),
                'download': True,
                'normalize': True,
                'augment': False
            },
            'logging': {
                'log_dir': os.path.join(self.temp_dir, 'logs'),
                'use_tensorboard': False,
                'use_wandb': False,
                'log_interval': 10,
                'save_model': False,
                'model_save_path': os.path.join(self.temp_dir, 'models')
            }
        }
        
        # Create necessary directories
        os.makedirs(self.config['data']['data_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['model_save_path'], exist_ok=True)
        
        self.trainer = QATTrainer(self.config)
    
    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        self.assertEqual(self.trainer.config, self.config)
        self.assertIsInstance(self.trainer.device, torch.device)
        self.assertIsInstance(self.trainer.db, MockDatabase)
    
    def test_prepare_data(self):
        """Test data preparation"""
        train_loader, test_loader = self.trainer.prepare_data()
        
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        
        # Test a batch
        for data, target in train_loader:
            self.assertEqual(data.shape[0], self.config['training']['batch_size'])
            self.assertEqual(data.shape[1:], (1, 28, 28))
            self.assertEqual(target.shape[0], self.config['training']['batch_size'])
            break
    
    def test_create_model(self):
        """Test model creation"""
        model = self.trainer.create_model()
        
        self.assertIsInstance(model, ModernQuantizedMLP)
        self.assertTrue(hasattr(model, 'qconfig'))
    
    def test_evaluate(self):
        """Test model evaluation"""
        # Create a simple model for testing
        model = self.trainer.create_model()
        train_loader, test_loader = self.trainer.prepare_data()
        criterion = torch.nn.CrossEntropyLoss()
        
        metrics = self.trainer.evaluate(model, test_loader, criterion)
        
        self.assertIn('val_loss', metrics)
        self.assertIn('val_accuracy', metrics)
        self.assertIsInstance(metrics['val_loss'], float)
        self.assertIsInstance(metrics['val_accuracy'], float)
        self.assertGreaterEqual(metrics['val_accuracy'], 0.0)
        self.assertLessEqual(metrics['val_accuracy'], 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up integration test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config for integration test
        self.config = {
            'model': {
                'input_size': 784,
                'hidden_sizes': [64],
                'output_size': 10,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 16,
                'test_batch_size': 50,
                'epochs': 1,
                'learning_rate': 0.01,
                'weight_decay': 1e-4,
                'scheduler_step_size': 5,
                'scheduler_gamma': 0.5
            },
            'quantization': {
                'backend': 'fbgemm',
                'observer_type': 'histogram',
                'quantize_activations': True,
                'quantize_weights': True,
                'calibration_samples': 50
            },
            'data': {
                'dataset': 'MNIST',
                'data_dir': os.path.join(self.temp_dir, 'data'),
                'download': True,
                'normalize': False,  # Skip normalization for faster testing
                'augment': False
            },
            'logging': {
                'log_dir': os.path.join(self.temp_dir, 'logs'),
                'use_tensorboard': False,
                'use_wandb': False,
                'log_interval': 5,
                'save_model': False,
                'model_save_path': os.path.join(self.temp_dir, 'models')
            }
        }
        
        # Create directories
        os.makedirs(self.config['data']['data_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['model_save_path'], exist_ok=True)
    
    def tearDown(self):
        """Clean up integration test"""
        shutil.rmtree(self.temp_dir)
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline"""
        trainer = QATTrainer(self.config)
        
        # This should complete without errors
        results = trainer.train()
        
        # Check results structure
        self.assertIn('final_accuracy', results)
        self.assertIn('model_size_mb', results)
        self.assertIn('training_time_seconds', results)
        self.assertIn('quantized_model', results)
        
        # Check result values
        self.assertGreaterEqual(results['final_accuracy'], 0.0)
        self.assertLessEqual(results['final_accuracy'], 1.0)
        self.assertGreater(results['model_size_mb'], 0.0)
        self.assertGreater(results['training_time_seconds'], 0.0)
        
        # Check quantized model
        self.assertIsInstance(results['quantized_model'], torch.nn.Module)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestMockDatabase))
    test_suite.addTest(unittest.makeSuite(TestModernQuantizedMLP))
    test_suite.addTest(unittest.makeSuite(TestQATTrainer))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Quantization-Aware Training Test Suite...")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
