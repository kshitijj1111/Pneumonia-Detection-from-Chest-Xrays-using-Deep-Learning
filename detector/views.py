import os
import io
import base64
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from skimage import filters, morphology
from scipy import ndimage as ndi
from tensorflow.keras.models import load_model
import gdown

# -------------------- Constants --------------------
HEALTHY_MEAN_PIXEL = 133.66
TARGET_SIZE = (512, 512)
MIN_OBJECT_SIZE = 300
IMG_SIZE = (224, 224)

# -------------------- Helper Functions --------------------
def read_gray(img_file):
    """Read image, convert to grayscale, resize."""
    img = Image.open(img_file).convert("L")
    return cv2.resize(np.array(img), TARGET_SIZE)

def preprocess_img(img):
    """Apply CLAHE and Gaussian blur."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return cv2.GaussianBlur(img_clahe, (5, 5), 0)

def get_lung_mask(img):
    """Extract lung mask."""
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    opened_bool = opened.astype(bool)
    filled = ndi.binary_fill_holes(opened_bool)
    labeled, n = ndi.label(filled)
    mask = np.zeros_like(img, dtype=bool)

    if n > 0:
        sizes = np.bincount(labeled.ravel())
        if sizes.size > 1:
            largest = np.argsort(sizes[1:])[::-1][:2] + 1
            for lab in largest:
                mask[labeled == lab] = True

    mask = morphology.remove_small_objects(mask, min_size=500)
    return ndi.binary_fill_holes(mask)

def get_gradcam(img_array, model_instance):
    try:
        last_conv_layer = model_instance.get_layer("block5_conv4")
        grad_model = tf.keras.models.Model([model_instance.inputs], [last_conv_layer.output, model_instance.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-6
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.image.resize(heatmap[..., None], IMG_SIZE).numpy().astype(np.uint8)
        heatmap_color = tf.keras.preprocessing.image.array_to_img(tf.repeat(heatmap, 3, axis=2))
        buffer = io.BytesIO()
        heatmap_color.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print("Grad-CAM error:", e)
        return None

# -------------------- Model Loading --------------------
model = None

def get_model():
    global model
    if model is None:
        model_path = os.path.join(settings.BASE_DIR, "model", "vgg19_pneumonia_model.keras")

        # Auto-download if missing
        if not os.path.exists(model_path):
            print("Model not found locally. Downloading from Google Drive...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Google Drive file ID
            file_id = "1tqOpYn9VIeyfV79b1pY7SsDy-5We8zxq"
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

            if not os.path.exists(model_path):
                raise FileNotFoundError("Model download failed. Check Google Drive link or permissions.")

        model = load_model(model_path)
        print("✅ Model loaded successfully!")

    return model


# -------------------- Views --------------------
@csrf_exempt
def single_detection(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("xray_image")
        if not uploaded_file:
            return JsonResponse({"error": "No file uploaded"}, status=400)

        try:
            # Preprocess image
            img = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(img.resize(IMG_SIZE)).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            model_instance = get_model()
            pred = model_instance.predict(img_array)
            print("Prediction output:", pred)

            # Handle sigmoid vs softmax
            if pred.shape[1] == 1:  # sigmoid
                score = float(pred[0][0])
            else:  # softmax
                score = float(pred[0][1])  # Check class index for Pneumonia

            status = "Pneumonia Detected" if score >= 0.5 else "Normal"
            confidence = round(score * 100, 2) if score >= 0.5 else round((1 - score) * 100, 2)

            if score < 0.5:
                severity, info = "Low", "No significant signs of pneumonia."
            elif score < 0.75:
                severity, info = "Moderate", "Possible signs of pneumonia. Monitor closely."
            else:
                severity, info = "High", "Strong evidence of pneumonia. Seek medical help."

            # Optional Grad-CAM
            gradcam = get_gradcam(img_array, model_instance)

            return JsonResponse({
                "status": status,
                "confidence": confidence,
                "severity": severity,
                "info": info,
                "gradcam": gradcam,
            })

        except Exception as e:
            print("Detection error:", e)
            return JsonResponse({"error": f"Detection failed: {str(e)}"})

    return render(request, "detector/single_detection.html")

@csrf_exempt
def multi_detection(request):
    """Handle multiple X-ray uploads for progression analysis."""
    if request.method == "POST":
        images = request.FILES.getlist("xray_images")
        dates = request.POST.getlist("xray_dates")

        infection_rates, image_dates = [], []

        for i, img_file in enumerate(images):
            img = read_gray(img_file)
            img_p = preprocess_img(img)
            lung_mask = get_lung_mask(img_p)

            if lung_mask.sum() == 0:
                infection_rates.append(0)
                image_dates.append(dates[i])
                continue

            corrected = img_p.astype(float) - HEALTHY_MEAN_PIXEL
            corrected[~lung_mask] = 0.0
            corrected = np.clip(corrected, 0, 255)

            vals = corrected[lung_mask]
            thresh = (filters.threshold_otsu(vals)
                      if vals.size > 0 and vals.std() >= 1e-6
                      else np.mean(vals) + 0.5 * np.std(vals))

            raw_mask = corrected > thresh
            clean = morphology.remove_small_objects(raw_mask, min_size=MIN_OBJECT_SIZE)
            clean = ndi.binary_closing(clean, structure=np.ones((5, 5)))
            clean = ndi.binary_fill_holes(clean) & lung_mask

            infection_percent = 100.0 * (clean.sum() / lung_mask.sum())
            infection_rates.append(round(infection_percent, 2))
            image_dates.append(dates[i])

        return JsonResponse({"dates": image_dates, "infection_rates": infection_rates})

    return render(request, "detector/multi_detection.html")


