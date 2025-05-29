import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd

# -------------------- Ayarlar --------------------
image_size = 32
num_classes = 43
model_path = 'trafik_modeli.keras'

st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

# -------------------- Dil Seçimi --------------------
lang = st.selectbox("🌐 Select Language / Dil Seçiniz", options=["Türkçe", "English"])

# -------------------- Dil Sözlüğü --------------------
text = {
    "Türkçe": {
        "title": "🚦 Trafik Tabelası Tanıma Sistemi",
        "desc": "Bir trafik tabelası fotoğrafı yükleyin, model tahmin etsin.",
        "upload": "📤 Bir trafik tabelası resmi yükleyin",
        "uploaded": "🖼️ Yüklenen Görsel",
        "predict": "✅ Tahmin Sonucu",
        "label": "**Etiket:**",
        "confidence": "**Güven:**",
        "top5": "🔍 En İyi 5 Tahmin",
        "error": "Model yüklenemedi. 'trafik_modeli.keras' dosyasını kontrol edin."
    },
    "English": {
        "title": "🚦 Traffic Sign Recognition System",
        "desc": "Upload a traffic sign image and let the model predict it.",
        "upload": "📤 Upload a traffic sign image",
        "uploaded": "🖼️ Uploaded Image",
        "predict": "✅ Prediction Result",
        "label": "**Label:**",
        "confidence": "**Confidence:**",
        "top5": "🔍 Top 5 Predictions",
        "error": "Model could not be loaded. Make sure 'trafik_modeli.keras' is in the directory."
    }
}

# -------------------- Başlık ve Açıklama --------------------
st.markdown(f"<h1 style='text-align: center;'>{text[lang]['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center;'>{text[lang]['desc']}</p>", unsafe_allow_html=True)

# -------------------- Model Yükleme --------------------
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"{text[lang]['error']} ({e})")
    st.stop()

# -------------------- Sınıf Adları --------------------
class_names = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited',
    17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# -------------------- Dosya Yükleme --------------------
uploaded_file = st.file_uploader(text[lang]['upload'], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption=text[lang]['uploaded'], use_column_width=True)

    # OpenCV ile işle (train işlemiyle birebir aynı)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR formatında okur
    img = cv2.resize(img, (image_size, image_size))   # (32,32)
    img = img / 255.0                                 # normalize
    img = np.expand_dims(img, axis=0)                 # (1,32,32,3)

    # -------------------- Tahmin --------------------
    predictions = model.predict(img)
    predicted_class_id = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100
    predicted_class_name = class_names.get(predicted_class_id, f"Unknown Class: {predicted_class_id}")

    st.markdown(f"### {text[lang]['predict']}")
    st.success(f"{text[lang]['label']} {predicted_class_name}")
    st.info(f"{text[lang]['confidence']} {confidence:.2f}%")

    # -------------------- Top 5 --------------------
    st.markdown("---")
    st.markdown(f"### {text[lang]['top5']}")

    df_predictions = pd.DataFrame({
        'Class ID': list(range(num_classes)),
        'Class Name': [class_names.get(i, f"Unknown Class: {i}") for i in range(num_classes)],
        'Confidence (%)': (predictions[0] * 100).tolist()
    }).sort_values(by='Confidence (%)', ascending=False)

    def highlight_top(s):
        return ['background-color: #d0f0c0' if i == 0 else '' for i in range(len(s))]

    st.dataframe(df_predictions.head(5).style
                 .format({'Confidence (%)': "{:.2f}%"})
                 .apply(highlight_top, axis=0))
