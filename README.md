# 🧠 MuSER-NLP: Multimodal Stress and Emotion Recognition

A real-time AI system that detects early signs of **stress**, **mental fatigue**, and **emotional states** by combining audio, facial, and textual (NLP) data using deep learning and unsupervised anomaly detection techniques.

---

## 🌟 Features

✅ Multimodal input support: audio + facial expression + text  
✅ Real-time stress detection using CNN-LSTM & Transformer models  
✅ Emotion classification trained on FER2013, RAVDESS, and transcript data  
✅ PCA + K-Means clustering for early anomaly detection in stress patterns  
✅ On-device ML optimization for edge deployment  

---

## 🧠 Tech Stack

| Component | Technology |
|----------|-------------|
| Text Analysis (NLP) | BERT, Transformers, NLTK, spaCy |
| Audio Emotion Recognition | MFCC + CNN-LSTM (RAVDESS dataset) |
| Facial Emotion Recognition | CNN on FER2013 dataset |
| Unsupervised Detection | PCA + K-Means clustering |
| Model Fusion | Late fusion via confidence voting |
| Deployment | PyTorch, TensorFlow, Streamlit |

---

## 📈 Model Performance

| Task | Accuracy |
|------|----------|
| Emotion Classification (text/audio/facial) | **91.3%** |
| Stress Detection Sensitivity | **+26%** (via PCA + K-Means) |

---

## 🔧 How It Works

1. **Input Sources:**
   - Facial image (from webcam/video)
   - Audio input (speech sample)
   - Text transcript (typed or extracted)

2. **Processing:**
   - Text → tokenized and passed to Transformer (e.g., BERT)
   - Audio → converted to MFCC → fed to CNN-LSTM
   - Image → processed via CNN for facial emotion features

3. **Fusion & Prediction:**
   - All features are fused
   - Final emotion/stress level is predicted
   - PCA + K-Means used to detect outliers for early stress markers

---

## 🚀 Getting Started

```bash
git clone https://github.com/amruthabhattttt/MuSER-Stress-NLP.git
cd MuSER-Stress-NLP
pip install -r requirements.txt
python main.py
