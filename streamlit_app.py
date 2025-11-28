import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim
from PIL import Image
import numpy as np

# ----------------------------------------------------
# CONFIGURACIÓN STREAMLIT
# ----------------------------------------------------
st.set_page_config(page_title="Detección Temprana de Roya", layout="centered")

st.title("🟢 Detección Temprana de Roya con Modelo Temporal")
st.write("Sube **dos imágenes de una misma hoja en dos semanas distintas** para analizar la evolución temporal.")

# ----------------------------------------------------
# ARQUITECTURA DEL MODELO — TemporalRoyaNet (CORRECTA)
# ----------------------------------------------------
class TemporalRoyaNet(nn.Module):
    def __init__(self, pooled_h=8, pooled_w=8, lstm_hidden=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((pooled_h, pooled_w))
        )

        self.pooled_h = pooled_h
        self.pooled_w = pooled_w
        self.feature_channels = 64
        lstm_input_size = self.feature_channels * pooled_h * pooled_w

        self.temporal = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            batch_first=True
        )

        self.decoder = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.size()
        feats = []

        for t in range(T):
            f = self.encoder(x_seq[:, t])
            feats.append(f.view(B, -1))

        feats = torch.stack(feats, dim=1)
        out, _ = self.temporal(feats)
        last = out[:, -1, :]
        return self.decoder(last)


# ----------------------------------------------------
# CARGA DEL MODELO ENTRENADO
# ----------------------------------------------------
@st.cache_resource
def load_model(path="temporal_royanet_ssim_model.pth"):
    try:
        model = TemporalRoyaNet()
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error cargando el modelo: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# ----------------------------------------------------
# TRANSFORMACIONES
# ----------------------------------------------------
transform_gray = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Grayscale()
])

# ----------------------------------------------------
# FUNCIÓN PARA ANALIZAR DOS IMÁGENES TEMPORALES
# ----------------------------------------------------
def analyze_temporal(img_fixed, img_moving, model):
    img1 = transform_gray(img_fixed)
    img2 = transform_gray(img_moving)

    x_seq = torch.stack([img1, img2], dim=0).unsqueeze(0)  # (1,2,1,256,256)

    with torch.no_grad():
        pred = model(x_seq).item()
        ssim_value = ssim(img1.unsqueeze(0), img2.unsqueeze(0)).item()
        variation = 1 - ssim_value

    return pred, variation


# ----------------------------------------------------
# INTERPRETACIÓN DE VARIACIÓN
# ----------------------------------------------------
def interpret_variation(variation):
    if variation < 0.20:
        return "SIN ROYA", "🟢"
    elif variation < 0.45:
        return "POSIBLE ROYA TEMPRANA", "🟡"
    else:
        return "ROYA DETECTADA", "🔴"


# ----------------------------------------------------
# UI PARA SUBIR IMÁGENES
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    img1_uploaded = st.file_uploader("📤 Imagen Semana X", type=["png", "jpg", "jpeg"])

with col2:
    img2_uploaded = st.file_uploader("📤 Imagen Semana Y", type=["png", "jpg", "jpeg"])

st.markdown("---")

if img1_uploaded and img2_uploaded:

    img1 = Image.open(img1_uploaded).convert("RGB")
    img2 = Image.open(img2_uploaded).convert("RGB")

    colA, colB = st.columns(2)

    with colA:
        st.image(img1, caption="Semana X (Fixed)")
    with colB:
        st.image(img2, caption="Semana Y (Moving)")

    st.markdown("---")

    if st.button("🔍 Analizar Roya"):
        pred, variation = analyze_temporal(img1, img2, model)
        label, emoji = interpret_variation(variation)

        st.subheader("📌 Resultado del Análisis Temporal")
        st.metric("Variación (1 − SSIM)", f"{variation:.4f}")
        st.metric("Diagnóstico", f"{emoji} {label}")

else:
    st.info("Sube dos imágenes para comenzar el análisis.")
