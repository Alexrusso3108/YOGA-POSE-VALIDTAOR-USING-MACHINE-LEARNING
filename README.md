# 🧘‍♂️ Yoga Pose Validator using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)  
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

## 📌 Overview

The **Yoga Pose Validator** is a machine learning-based project that evaluates yoga postures using computer vision techniques. The system leverages **MediaPipe, OpenCV, and TensorFlow** to analyze and classify yoga poses in real-time, providing feedback to users on their posture correctness.

## ✨ Features
- 🔍 **Real-time pose detection** using OpenCV and MediaPipe
- 🎯 **Pose classification** with a pre-trained TensorFlow model
- 📊 **Accuracy feedback** based on key body points alignment
- 📹 **Supports webcam input** for real-time analysis
- 💾 **Expandable model** to include more yoga poses

---

## 🛠️ Installation

### 1️⃣ Clone the Repository
```sh
 git clone https://github.com/Alexrusso3108/YOGA-POSE-VALIDATOR-USING-MACHINE-LEARNING.git
 cd YOGA-POSE-VALIDATOR-USING-MACHINE-LEARNING
```

### 2️⃣ Create a Virtual Environment (Recommended)
```sh
 python -m venv yoga_env
 source yoga_env/bin/activate   # On macOS/Linux
 yoga_env\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```sh
 pip install -r requirements.txt
```

---

## 🚀 Usage

### 1️⃣ Run the Yoga Pose Validator
```sh
 python main.py
```

### 2️⃣ Webcam Input
- Ensure your webcam is connected.
- The system will detect and validate your yoga pose in real-time.

### 3️⃣ Adding New Poses
- Collect images of the new pose.
- Train the TensorFlow model with the new dataset.
- Update the pose classification logic.

---

## 📂 Project Structure
```
📁 yoga-pose-validator
│── 📂 models                # Pre-trained ML models
│── 📂 dataset               # Training images/videos
│── 📂 scripts               # Helper scripts for data preprocessing
│── main.py                  # Main application
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

---

## 🧠 Technologies Used
- **Python** 🐍
- **OpenCV** 🎥
- **MediaPipe** 🤖
- **TensorFlow/Keras** 🧠
- **NumPy & Pandas** 📊

---

## 🤝 Contributing
Want to improve the project? Feel free to fork and submit a PR! 🚀
```sh
 git clone https://github.com/your-username/YOGA-POSE-VALIDATOR-USING-MACHINE-LEARNING.git
 git checkout -b feature-branch
 git commit -m "Your feature description"
 git push origin feature-branch
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Show Your Support!
If you like this project, please **star** ⭐ the repository and share it with others!

---

🔗 **GitHub Repo:** [Yoga Pose Validator](https://github.com/Alexrusso3108/YOGA-POSE-VALIDATOR-USING-MACHINE-LEARNING)
