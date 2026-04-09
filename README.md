# 🧠 NanoAI

A minimal transformer-based language model built from scratch with PyTorch.

## 📁 Project Structure

NanoAI — data — train.txt (Training data), scripts — model.py (Transformer model architecture), train.py (Training script), generate.py (Text generation script), models (Saved model checkpoints), requirements.txt (Dependencies), README.md

## 🚀 Quick Start

### 1. Install dependencies

pip install -r requirements.txt

### 2. Train the model

cd scripts
python train.py

### 3. Generate text

cd scripts
python generate.py "the future of"

## 🏗️ Model Architecture

- **Type:** Transformer Encoder
- **Embedding size:** 128
- **Attention heads:** 4
- **Layers:** 2
- **Vocabulary:** 5000 words max
- **Parameters:** ~1.5M

## 📊 How It Works

1. Text is split into words and converted to numbers
2. The transformer learns patterns in the training data
3. At generation time, it predicts the next word given context
4. Words are sampled based on probability distribution

## 🛠️ Tech Stack

- Python 3.8+
- PyTorch 2.0+
- Pure transformer architecture (no external NLP libraries)

## 📝 License

MIT License — use freely!

## ⭐ Star this repo if you like it!
