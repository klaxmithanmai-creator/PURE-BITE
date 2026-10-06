# app.py
import os
# Suppress TensorFlow warnings/logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="PureBite – Food Purity AI", page_icon="🥗")

# ------------------- Helper Functions -------------------
def preprocess_image(uploaded_file):
    """Convert uploaded image to numpy array and resize for model input."""
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    arr = np.array(image)
    arr_resized = cv2.resize(arr, (224, 224)) / 255.0
    return np.expand_dims(arr_resized, axis=0)

def predict_tf(img_batch):
    """Predict using TensorFlow/Keras model if available."""
    try:
        model = tf.keras.models.load_model("models/food_purity_model.h5")
        preds = model.predict(img_batch)
        classes = ["Pure", "Contaminated"]
        return classes[np.argmax(preds)], float(np.max(preds) * 100)
    except Exception as e:
        raise RuntimeError(f"TensorFlow model error: {e}")

def predict_xgb(img_batch):
    """Predict using XGBoost model if available."""
    try:
        model = joblib.load("models/xgb_food_model.pkl")
        flat = img_batch.reshape(1, -1)
        return model.predict(flat)[0]
    except Exception as e:
        raise RuntimeError(f"XGBoost model error: {e}")

# ------------------- Tabs (Pages) -------------------
tab1, tab2, tab3 = st.tabs(["📘 About", "🔍 Detection", "📊 Insights"])

# ------------------- Page 1: About -------------------
with tab1:
    st.title("📘 About PureBite")
    st.write("""
    PureBite is a Smart Food Purity Detection System designed to ensure food safety.  
    It uses **Deep Learning (TensorFlow/Keras)** and **Machine Learning (XGBoost, Scikit-Learn)** models 
    to classify food images as **Pure** or **Contaminated**.
    
    ### Features
    - 🖼️ Image Upload & Preprocessing
    - 🤖 AI Model Predictions
    - 📊 Visual Insights
    
    ### Goal
    To provide a simple, reliable, and AI-powered solution for food quality monitoring.
    """)

# ------------------- Page 2: Detection -------------------
with tab2:
    st.title("🔍 Food Purity Detection")
    uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img_batch = preprocess_image(uploaded_file)
        try:
            pred_class, confidence = predict_tf(img_batch)
            st.success(f"Prediction: **{pred_class}** ({confidence:.2f}%)")
        except Exception as e1:
            st.warning(str(e1))
            try:
                pred_class = predict_xgb(img_batch)
                st.success(f"Prediction: **{pred_class}**")
            except Exception as e2:
                st.error(str(e2))
                st.info("Fallback: Demo mode → Random prediction")
                demo_pred = np.random.choice(["Pure", "Contaminated"])
                st.success(f"Prediction: **{demo_pred}** (Demo)")
    else:
        st.info("👆 Upload an image to start detection.")

# ------------------- Page 3: Insights -------------------
with tab3:
    st.title("📊 Insights & Analytics")
    st.write("Here are some example insights from model performance:")

    fig, ax = plt.subplots()
    sns.heatmap([[50, 5], [3, 42]], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pure", "Contaminated"],
                yticklabels=["Pure", "Contaminated"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.write("""
    ✅ High accuracy in detecting pure food.  
    ⚠️ Some false positives/negatives exist — continuous training improves results.  
    """)

# ------------------- Footer -------------------
st.caption("Made with ❤️ by Laxmi | Powered by Streamlit & AI")
