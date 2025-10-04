# Makefile for Quantization-Aware Training Project

.PHONY: help install test demo original ui clean train setup

# Default target
help:
	@echo "🧠 Quantization-Aware Training Project"
	@echo "======================================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup     - Install dependencies and create directories"
	@echo "  install   - Install Python dependencies only"
	@echo "  test      - Run test suite"
	@echo "  demo      - Run demo script"
	@echo "  original  - Run original implementation"
	@echo "  ui        - Launch Streamlit web UI"
	@echo "  train     - Run advanced trainer with visualizations"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make setup    # Complete setup"
	@echo "  make demo     # Quick demo"
	@echo "  make ui       # Launch web interface"

# Setup everything
setup:
	@echo "🚀 Setting up QAT project..."
	python setup.py --test --demo
	@echo "✅ Setup completed!"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Run tests
test:
	@echo "🧪 Running test suite..."
	python test_qat.py
	@echo "✅ Tests completed!"

# Run demo
demo:
	@echo "🚀 Running demo..."
	python demo.py
	@echo "✅ Demo completed!"

# Run original implementation
original:
	@echo "🧠 Running original implementation..."
	python 0155.py
	@echo "✅ Original implementation completed!"

# Launch web UI
ui:
	@echo "🌐 Launching Streamlit UI..."
	@echo "Open your browser to http://localhost:8501"
	streamlit run app.py

# Run advanced trainer
train:
	@echo "🚀 Running advanced trainer..."
	python qat_trainer.py --visualize
	@echo "✅ Training completed!"

# Clean up generated files
clean:
	@echo "🧹 Cleaning up..."
	rm -rf __pycache__/
	rm -rf *.pyc
	rm -rf logs/
	rm -rf models/
	rm -rf plots/
	rm -f *.log
	rm -f *.db
	@echo "✅ Cleanup completed!"

# Create directories
dirs:
	@echo "📁 Creating directories..."
	mkdir -p data logs models plots
	@echo "✅ Directories created!"
