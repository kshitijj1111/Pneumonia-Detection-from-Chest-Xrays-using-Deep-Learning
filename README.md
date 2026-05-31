# Pneumonia-Detection-from-Chest-Xrays-using-Deep-Learning

# 🫁 Pneumonia Detection Web App
### (Django + TensorFlow + VGG19)

A **Deep Learning–powered** web application built to detect **Pneumonia** from chest X-ray images. By combining a fine-tuned **VGG19** model with a robust **Django** backend, this project supports both **single** and **multiple image** predictions with an intuitive, responsive interface.

---

## 🔗 Quick Links

- **All Models (Drive):** [Google Drive](https://drive.google.com/drive/folders/1U46EsIHwJhOGTT6QQvoPfhkrfDRHd1Iw?usp=sharing)  
- **Dataset Used:** [Paul Mooney (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  

---

## 🚀 Features & Detection Modes

### 🩺 1. Single Detection Mode
Upload a **Chest X-ray**, and the model detects whether the patient has Pneumonia or not in real-time.

### 📈 2. Multi Detection Mode
Upload **multiple X-rays** (with dates), and the system generates a **graph** showing the increase or decrease in Pneumonia infection over time.
* **The Shadow Fix:** Initially, the system misinterpreted shadowed areas outside the lungs as infection zones. To counter this, the model calculates the average shadowed pixels from healthy lung X-rays and subtracts that baseline from the Pneumonia images. It produces highly believable and satisfactory tracking results.

---

## 🧠 Tech Stack & Dependencies

| Layer | Technology |
|:------|:------------|
| 🎨 **Frontend** | HTML, CSS, JavaScript |
| ⚙️ **Backend** | Django (Python) |
| 🧩 **Model** | VGG19 (Pretrained on ImageNet, fine-tuned for X-rays) |
| ☁️ **Deployment**| Ready for Render or any cloud platform |

### 📦 Major Python Dependencies

| Package | Description |
|:-----------|:---------------|
| `Django` | High-level Python web framework |
| `TensorFlow` & `Keras` | Deep Learning library and API for the detection model |
| `NumPy` & `Pandas` | Numerical computing and data manipulation |
| `OpenCV-Python` | Image processing and computer vision toolkit |
| `Matplotlib` | Visualization library used for generating infection graphs |
| `Pillow` | Image handling and conversion library |
| `Gunicorn` & `WhiteNoise` | WSGI HTTP server and static file serving for production |
| `python-dotenv` | Loads environment variables from `.env` files |

---

## 🧬 Model Performance & Dataset

**Dataset Summary**
| Dataset Type  | Normal Images | Pneumonia Images |
|---------------|---------------|------------------|
| Training      | 1341          | 3875             |
| Validation    | 8             | 8                |
| Testing       | 234           | 390              |

**Models Trained (30 Epochs Each)**
| Model            | Accuracy | Notes |
|------------------|-----------|-------|
| **VGG-19** | **91.67%**| ✅ **Chosen for deployment** (Best for X-ray related ML) |
| **DenseNet121** | 90.38%    | - |
| **ResNet152** | 73.88%    | - |
| **EfficientNet-B3** | 62.50% | ⚠️ Early Stopping |

---

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Pneumonia-Detection-Django.git
   cd Pneumonia-Detection-Django

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run migrations:
   ```bash
   python manage.py migrate

5. Start the server:
   ```bash
   python manage.py runserver

---
## 💀 Developer Notes & Struggles

Building this project was a massive learning experience, complete with a few developer nightmares:

* **The Medical Data Struggle:** Medical-related datasets can be incredibly sparse, inconsistent, and an absolute pain to work with. Cleaning and processing this data was half the battle.
* **The Dependency Hell:** I initially built this without setting up a virtual environment and lost track of my dependencies. While trying to fix it, I ran into the infamous `"List object has no attribute 'shape'"` error—a classic NumPy/TensorFlow mismatch where the model was getting raw Python lists instead of formatted arrays. After hours of trial and error, I ended up freezing my global environment into the `requirements.txt`. Now everything works... somehow.
