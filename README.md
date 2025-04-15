# Gesturon 🖐️🤖

> Real-time gesture-to-speech translator powered by deep learning and computer vision.

**Gesturon** is an AI-powered system that detects hand gestures in real-time using a webcam, classifies them using a custom-trained LSTM model, and speaks them out loud. Built entirely in Python, it bridges silence and speech with the wave of a hand.

---

## 💡 Features

- 🖐️ Real-time hand tracking with Mediapipe
- 🎯 Personalized gesture training on your own hand movements
- 🧠 Sequence-based classification with LSTM
- 🗣️ Text-to-speech output using pyttsx3
- 🖥️ Built for desktop using OpenCV and TensorFlow

---

## 📂 Tech Stack

- Python 3.8
- OpenCV
- Mediapipe
- TensorFlow (LSTM)
- pyttsx3 (text-to-speech)
- Numpy, Scikit-learn

---

## 🚀 How It Works

1. **Collect Data**: Record gestures like "hello", "thanks", "yes", and "no" with `collect_landmark_data.py`
2. **Train Model**: Run `train_model.py` to train your custom gesture classifier
3. **Real-Time Prediction**: Launch `real_time_predictor.py` to detect gestures and speak them out loud in real time

---

## 📦 Setup Instructions

```bash
# Create and activate environment
conda create -n gesturon python=3.8
conda activate gesturon

# Install dependencies
pip install -r requirements.txt
