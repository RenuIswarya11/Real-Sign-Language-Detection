# ✋ Real-Time Sign Language Detection

An **AI-powered hand gesture recognition** system that detects and classifies **sign language gestures in real-time** using **MediaPipe** and **TensorFlow/Keras**.  
This lightweight application processes live webcam input and identifies gestures such as **Hello**, **Thank You**, **Yes**, **No**, and **I Love You**.

---

## 🚀 Features

- Capture and process hand gestures using **MediaPipe Hand Tracking**  
- Train a custom gesture classification model using **TensorFlow/Keras**
- Detect gestures from **live webcam feed** in real-time  
- Lightweight and easy to customize (no large datasets required)
- Simple and clean code structure for beginners  

---

## 🛠 Tech Stack

- **Framework & Language**: Python (3.9+)
- **Computer Vision**: [OpenCV], [MediaPipe]
- **Deep Learning**: [TensorFlow / Keras] 
- **Data Processing**: [NumPy], [scikit-learn]
- **Visualization**: Matplotlib 

---

## 📂 Project Structure

Real-Time Sign Language Detection/
│
├── data/
│ ├── label_map.json    # Label mapping for gesture classes
│ ├── X.npy     # Landmark feature data
│ └── y.npy     # Encoded gesture labels
│
├── dataset/
│ ├── hello/    # Captured gesture images
│ ├── i_love_you/
│ ├── no/
│ ├── thank_you/
│ └── yes/
│
├── models/
│ ├── best_model.h5     # Best performing trained model
│ ├── final_model.h5    # Final trained model
│ ├── scaler_mean.npy   # Scaler mean values (for normalization)
│ └── scaler_scale.npy  # Scaler scale values (for normalization)
│
├── source/
│ ├── capture_images.py     # Capture gesture images through webcam
│ ├── extract_landmarks.py  # Extract hand landmarks using MediaPipe
│ ├── realtimedetection.py  # Run real-time gesture recognition
│ └── train.py  # Train the sign language detection model
│
├── venv/   # Virtual environment (excluded from repo)
├── .gitignore  # Ignored files
├── README.md   # Project documentation
└── requirements.txt # Required dependencies   

---

## 🖼️ Project Preview

Here’s a quick look at the **Real Time Sign Language Detection** app in action!  
It captures real-time hand gestures via webcam and predicts corresponding signs such as *Hello*, *Thank You*, *Yes*, *No*, and *I Love You* with a trained deep learning model.

<p align="center">
  <img src="docs/sign1.png" alt="Real Sign Language Detection Preview" width="800">
</p>


<p align="center">
   <img src="docs/sign3.png" alt="Real Sign Language Detection Demo" width="800">
 </p>




