#!/usr/bin/env python3
"""
Streamlit UI for Quantization-Aware Training
Modern web interface for QAT model training and monitoring
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import yaml
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="QAT Training Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .error-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class QATDashboard:
    """Main dashboard class for QAT training interface"""
    
    def __init__(self):
        self.db_path = "qat_results.db"
        self.config_path = "config.yaml"
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            st.error(f"Configuration file {self.config_path} not found!")
            return None
    
    def get_experiment_history(self):
        """Get experiment history from database"""
        try:
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
        except sqlite3.OperationalError:
            return pd.DataFrame()
    
    def get_training_logs(self, experiment_id):
        """Get training logs for a specific experiment"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT * FROM training_logs 
                WHERE experiment_id = ?
                ORDER BY epoch
            ''', conn, params=(experiment_id,))
            conn.close()
            return df
        except sqlite3.OperationalError:
            return pd.DataFrame()
    
    def run_training(self, config_updates):
        """Run training with updated configuration"""
        # Update config file
        config = self.load_config()
        if config is None:
            return False
        
        # Apply updates
        for key, value in config_updates.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Run training
        try:
            result = subprocess.run([
                sys.executable, 'qat_trainer.py', 
                '--config', self.config_path, '--visualize'
            ], capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                st.success("Training completed successfully!")
                return True
            else:
                st.error(f"Training failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            st.error("Training timed out!")
            return False
        except Exception as e:
            st.error(f"Error running training: {str(e)}")
            return False

def main():
    """Main Streamlit application"""
    dashboard = QATDashboard()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Quantization-Aware Training Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Overview", "⚙️ Configuration", "🚀 Training", "📊 Results", "📈 Analytics"]
    )
    
    if page == "🏠 Overview":
        show_overview(dashboard)
    elif page == "⚙️ Configuration":
        show_configuration(dashboard)
    elif page == "🚀 Training":
        show_training(dashboard)
    elif page == "📊 Results":
        show_results(dashboard)
    elif page == "📈 Analytics":
        show_analytics(dashboard)

def show_overview(dashboard):
    """Show overview page"""
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Project Description")
        st.markdown("""
        This project implements **Quantization-Aware Training (QAT)** using modern PyTorch 2.0+ APIs.
        
        **Key Features:**
        - 🧠 Modern neural network architecture with BatchNorm and Dropout
        - 📊 Comprehensive logging and metrics tracking
        - 🗄️ Mock database for experiment management
        - 📈 Real-time training visualization
        - ⚙️ Configurable training parameters
        - 🎯 Quantization simulation during training
        
        **Benefits of QAT:**
        - Higher accuracy compared to post-training quantization
        - Better suited for edge devices and mobile applications
        - Reduced model size and inference time
        - Maintains performance with int8 precision
        """)
    
    with col2:
        st.subheader("📊 Recent Experiments")
        history_df = dashboard.get_experiment_history()
        
        if not history_df.empty:
            # Show latest experiment
            latest = history_df.iloc[0]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Latest Accuracy", f"{latest['final_accuracy']:.2%}")
            with col_b:
                st.metric("Model Size", f"{latest['model_size_mb']:.2f} MB")
            with col_c:
                st.metric("Training Time", f"{latest['training_time_seconds']:.1f}s")
            
            # Show experiment history table
            st.subheader("Experiment History")
            display_df = history_df[['timestamp', 'quantization_backend', 'final_accuracy', 'model_size_mb']].copy()
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            display_df.columns = ['Timestamp', 'Backend', 'Accuracy', 'Size (MB)']
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No experiments found. Start training to see results here!")

def show_configuration(dashboard):
    """Show configuration page"""
    st.header("⚙️ Configuration Settings")
    
    config = dashboard.load_config()
    if config is None:
        return
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Model", "🎯 Training", "📦 Quantization", "📊 Logging"])
    
    with tab1:
        st.subheader("Model Architecture")
        col1, col2 = st.columns(2)
        
        with col1:
            input_size = st.number_input("Input Size", value=config['model']['input_size'], min_value=1)
            hidden_size_1 = st.number_input("Hidden Layer 1 Size", value=config['model']['hidden_sizes'][0], min_value=1)
            hidden_size_2 = st.number_input("Hidden Layer 2 Size", value=config['model']['hidden_sizes'][1], min_value=1)
        
        with col2:
            output_size = st.number_input("Output Size", value=config['model']['output_size'], min_value=1)
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, config['model']['dropout_rate'], 0.01)
    
    with tab2:
        st.subheader("Training Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], 
                                    index=[32, 64, 128, 256].index(config['training']['batch_size']))
            epochs = st.number_input("Epochs", value=config['training']['epochs'], min_value=1, max_value=100)
            learning_rate = st.number_input("Learning Rate", value=config['training']['learning_rate'], 
                                          min_value=1e-5, max_value=1.0, format="%.5f")
        
        with col2:
            weight_decay = st.number_input("Weight Decay", value=config['training']['weight_decay'], 
                                         min_value=0.0, max_value=1e-2, format="%.5f")
            scheduler_step = st.number_input("Scheduler Step Size", value=config['training']['scheduler_step_size'], 
                                           min_value=1, max_value=20)
            scheduler_gamma = st.slider("Scheduler Gamma", 0.1, 0.9, config['training']['scheduler_gamma'], 0.1)
    
    with tab3:
        st.subheader("Quantization Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            backend = st.selectbox("Quantization Backend", ["fbgemm", "qnnpack"], 
                                 index=["fbgemm", "qnnpack"].index(config['quantization']['backend']))
            observer_type = st.selectbox("Observer Type", ["histogram", "minmax"], 
                                       index=["histogram", "minmax"].index(config['quantization']['observer_type']))
        
        with col2:
            quantize_activations = st.checkbox("Quantize Activations", value=config['quantization']['quantize_activations'])
            quantize_weights = st.checkbox("Quantize Weights", value=config['quantization']['quantize_weights'])
            calibration_samples = st.number_input("Calibration Samples", value=config['quantization']['calibration_samples'], 
                                               min_value=10, max_value=1000)
    
    with tab4:
        st.subheader("Logging and Output")
        col1, col2 = st.columns(2)
        
        with col1:
            use_tensorboard = st.checkbox("Use TensorBoard", value=config['logging']['use_tensorboard'])
            use_wandb = st.checkbox("Use Weights & Biases", value=config['logging']['use_wandb'])
            log_interval = st.number_input("Log Interval", value=config['logging']['log_interval'], min_value=1, max_value=1000)
        
        with col2:
            save_model = st.checkbox("Save Model", value=config['logging']['save_model'])
            model_save_path = st.text_input("Model Save Path", value=config['logging']['model_save_path'])
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        config_updates = {
            'model.input_size': input_size,
            'model.hidden_sizes': [hidden_size_1, hidden_size_2],
            'model.output_size': output_size,
            'model.dropout_rate': dropout_rate,
            'training.batch_size': batch_size,
            'training.epochs': epochs,
            'training.learning_rate': learning_rate,
            'training.weight_decay': weight_decay,
            'training.scheduler_step_size': scheduler_step,
            'training.scheduler_gamma': scheduler_gamma,
            'quantization.backend': backend,
            'quantization.observer_type': observer_type,
            'quantization.quantize_activations': quantize_activations,
            'quantization.quantize_weights': quantize_weights,
            'quantization.calibration_samples': calibration_samples,
            'logging.use_tensorboard': use_tensorboard,
            'logging.use_wandb': use_wandb,
            'logging.log_interval': log_interval,
            'logging.save_model': save_model,
            'logging.model_save_path': model_save_path
        }
        
        # Update config
        for key, value in config_updates.items():
            keys = key.split('.')
            current = config
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        
        # Save to file
        with open(dashboard.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        st.success("Configuration saved successfully!")

def show_training(dashboard):
    """Show training page"""
    st.header("🚀 Start Training")
    
    config = dashboard.load_config()
    if config is None:
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration Summary")
        
        # Display current configuration
        config_summary = {
            "Model": f"{config['model']['hidden_sizes']} → {config['model']['output_size']}",
            "Training": f"{config['training']['epochs']} epochs, batch size {config['training']['batch_size']}",
            "Quantization": f"Backend: {config['quantization']['backend']}",
            "Learning Rate": f"{config['training']['learning_rate']:.5f}"
        }
        
        for key, value in config_summary.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("Training in progress..."):
                success = dashboard.run_training({})
                if success:
                    st.balloons()
                    st.success("Training completed! Check the Results page for details.")
                else:
                    st.error("Training failed. Check the logs for details.")
        
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
    
    # Training status
    st.subheader("📊 Training Status")
    history_df = dashboard.get_experiment_history()
    
    if not history_df.empty:
        latest = history_df.iloc[0]
        
        # Create status indicators
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Latest Accuracy", f"{latest['final_accuracy']:.2%}")
        with col_b:
            st.metric("Model Size", f"{latest['model_size_mb']:.2f} MB")
        with col_c:
            st.metric("Training Time", f"{latest['training_time_seconds']:.1f}s")
        with col_d:
            st.metric("Total Experiments", len(history_df))
    else:
        st.info("No training experiments found. Start your first training session!")

def show_results(dashboard):
    """Show results page"""
    st.header("📊 Training Results")
    
    history_df = dashboard.get_experiment_history()
    
    if history_df.empty:
        st.info("No experiments found. Start training to see results here!")
        return
    
    # Experiment selector
    st.subheader("Select Experiment")
    experiment_options = []
    for idx, row in history_df.iterrows():
        timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
        option = f"Experiment {row['id']} - {timestamp} (Acc: {row['final_accuracy']:.2%})"
        experiment_options.append((row['id'], option))
    
    selected_id = st.selectbox("Choose experiment:", experiment_options, format_func=lambda x: x[1])
    experiment_id = selected_id[0]
    
    # Get training logs for selected experiment
    logs_df = dashboard.get_training_logs(experiment_id)
    
    if logs_df.empty:
        st.warning("No training logs found for this experiment.")
        return
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Accuracy", f"{logs_df['val_accuracy'].iloc[-1]:.2%}")
    with col2:
        st.metric("Final Loss", f"{logs_df['val_loss'].iloc[-1]:.4f}")
    with col3:
        st.metric("Best Accuracy", f"{logs_df['val_accuracy'].max():.2%}")
    with col4:
        st.metric("Total Epochs", len(logs_df))
    
    # Training progress plots
    st.subheader("📈 Training Progress")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Loss Over Time', 'Accuracy Over Time', 'Loss Comparison', 'Accuracy Comparison'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss over time
    fig.add_trace(
        go.Scatter(x=logs_df['epoch'], y=logs_df['train_loss'], 
                  name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=logs_df['epoch'], y=logs_df['val_loss'], 
                  name='Val Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Accuracy over time
    fig.add_trace(
        go.Scatter(x=logs_df['epoch'], y=logs_df['train_accuracy'], 
                  name='Train Accuracy', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=logs_df['epoch'], y=logs_df['val_accuracy'], 
                  name='Val Accuracy', line=dict(color='red')),
        row=1, col=2
    )
    
    # Loss comparison (bar chart)
    fig.add_trace(
        go.Bar(x=['Train', 'Validation'], 
               y=[logs_df['train_loss'].iloc[-1], logs_df['val_loss'].iloc[-1]],
               name='Final Loss'),
        row=2, col=1
    )
    
    # Accuracy comparison (bar chart)
    fig.add_trace(
        go.Bar(x=['Train', 'Validation'], 
               y=[logs_df['train_accuracy'].iloc[-1], logs_df['val_accuracy'].iloc[-1]],
               name='Final Accuracy'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Training Metrics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Metrics")
    st.dataframe(logs_df, use_container_width=True)

def show_analytics(dashboard):
    """Show analytics page"""
    st.header("📈 Analytics & Insights")
    
    history_df = dashboard.get_experiment_history()
    
    if history_df.empty:
        st.info("No experiments found. Start training to see analytics!")
        return
    
    # Overall statistics
    st.subheader("📊 Overall Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(history_df))
    with col2:
        st.metric("Best Accuracy", f"{history_df['final_accuracy'].max():.2%}")
    with col3:
        st.metric("Average Accuracy", f"{history_df['final_accuracy'].mean():.2%}")
    with col4:
        st.metric("Average Model Size", f"{history_df['model_size_mb'].mean():.2f} MB")
    
    # Performance trends
    st.subheader("📈 Performance Trends")
    
    # Convert timestamp to datetime
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    # Create trend plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Over Time', 'Model Size Over Time', 
                       'Training Time Over Time', 'Accuracy Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy over time
    fig.add_trace(
        go.Scatter(x=history_df['timestamp'], y=history_df['final_accuracy'], 
                  mode='lines+markers', name='Accuracy'),
        row=1, col=1
    )
    
    # Model size over time
    fig.add_trace(
        go.Scatter(x=history_df['timestamp'], y=history_df['model_size_mb'], 
                  mode='lines+markers', name='Model Size'),
        row=1, col=2
    )
    
    # Training time over time
    fig.add_trace(
        go.Scatter(x=history_df['timestamp'], y=history_df['training_time_seconds'], 
                  mode='lines+markers', name='Training Time'),
        row=2, col=1
    )
    
    # Accuracy distribution
    fig.add_trace(
        go.Histogram(x=history_df['final_accuracy'], name='Accuracy Distribution'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Experiment Analytics")
    st.plotly_chart(fig, use_container_width=True)
    
    # Backend comparison
    st.subheader("🔧 Backend Comparison")
    
    backend_stats = history_df.groupby('quantization_backend').agg({
        'final_accuracy': ['mean', 'std', 'count'],
        'model_size_mb': 'mean',
        'training_time_seconds': 'mean'
    }).round(4)
    
    st.dataframe(backend_stats, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    best_experiment = history_df.loc[history_df['final_accuracy'].idxmax()]
    
    st.markdown(f"""
    **Best Performing Experiment:**
    - **Accuracy:** {best_experiment['final_accuracy']:.2%}
    - **Backend:** {best_experiment['quantization_backend']}
    - **Model Size:** {best_experiment['model_size_mb']:.2f} MB
    - **Training Time:** {best_experiment['training_time_seconds']:.1f} seconds
    
    **Insights:**
    - Average accuracy across all experiments: {history_df['final_accuracy'].mean():.2%}
    - Most experiments use {history_df['quantization_backend'].mode().iloc[0]} backend
    - Model size ranges from {history_df['model_size_mb'].min():.2f} to {history_df['model_size_mb'].max():.2f} MB
    """)

if __name__ == "__main__":
    main()
