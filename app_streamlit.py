
# app_streamlit.py
import streamlit as st
from PIL import Image
import joblib, os
import numpy as np
import torch
from torchvision import models, transforms
from gtts import gTTS
import wikipedia

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "svc_pipeline.joblib")
TMP_AUDIO = os.path.join(BASE_DIR, "resp.mp3")

data = joblib.load(MODEL_PATH)
clf = data['model']
labels = data['labels']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feat_model = models.mobilenet_v2(pretrained=True).features.to(device)
feat_model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def extract_feature_pil(img_pil):
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feat_model(x)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1)).squeeze().cpu().numpy()
    return feat.reshape(1,-1)

st.title("Species-like Visual Assistant (Demo)")

uploaded = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)
    feat = extract_feature_pil(img)
    probs = clf.predict_proba(feat)[0]
    idx = int(probs.argmax())
    label = labels[idx]
    conf = probs[idx]
    st.markdown(f"**Prediction:** `{label}`  —  **Confidence:** {conf:.2f}")

    summary = ""
    page_url = ""
    try:
        wikipedia.set_lang("ar")
        summary = wikipedia.summary(label, sentences=2)
        page_url = wikipedia.page(label).url
    except Exception:
        try:
            wikipedia.set_lang("en")
            summary = wikipedia.summary(label, sentences=2)
            page_url = wikipedia.page(label).url
        except Exception:
            summary = "لم أستطع العثور على ملخص لهذا الكائن."

    st.write("**Summary:**")
    st.write(summary)
    if page_url:
        st.write("Source:", page_url)

    if st.button("Play audio"):
        lang = 'ar' if wikipedia.get_lang()=='ar' else 'en'
        tts = gTTS(summary, lang=lang)
        tts.save(TMP_AUDIO)
        audio_bytes = open(TMP_AUDIO, "rb").read()
        st.audio(audio_bytes, format="audio/mp3")
