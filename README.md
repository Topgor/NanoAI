<div align="center">

# 🧬 NanoAI

### Transformer Language Model Trained from Scratch on CPU

**No GPU. No Cloud. No Pretrained Weights. Just Pure Code.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Topgor/NanoAI?style=social)](https://github.com/Topgor/NanoAI)

<img src="https://img.shields.io/badge/Parameters-13M-purple?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Trained_on-CPU_Only-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/OS-Kali_Linux-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Model_Size-50_MB-green?style=for-the-badge"/>

---

*A 13-million parameter transformer language model, written from scratch in PyTorch, 
trained entirely on a laptop CPU running Kali Linux. No GPU was used at any point.*

</div>

---

## 🔥 Why This Project Exists

Everyone says you need expensive GPUs to train AI. **I proved them wrong.**

I built a transformer from scratch — every attention head, every layer norm, every training loop — 
and trained it on a regular laptop CPU. The model generates coherent Russian text after just 50 epochs.

**This is Day 1. The journey to 1 billion tokens starts here.**

---

## 📊 Training Progress

| Epoch | Loss | Time/Epoch | Sample Output |
|-------|------|------------|---------------|
| 1 | 3.247 | 503s | `йцукенгш...` (random noise) |
| 10 | 0.038 | 498s | `Я NanoAI компактный искусственный интеллект` |
| 20 | 0.024 | 501s | `Нейросеть это математическая модель...` |
| 30 | 0.021 | 503s | `Искусственный интеллект это система способная решать задачи требующие человеческого интеллекта` |
| 50 | TBD | TBD | *Training in progress...* |

> **Loss dropped from 3.247 → 0.021 in 30 epochs.** The model went from random characters to generating coherent Russian sentences about AI, programming, and science.

---

## 🧠 Architecture

