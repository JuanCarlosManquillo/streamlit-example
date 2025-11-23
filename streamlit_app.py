import streamlit as st
import numpy as np
from PIL import Image
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim

# ----------------------------
#   Cargar modelo (ajusta esto)
# ----------------------------
# @st.cache_resource
# def load_model():
#     model = torch.load("modelo_fase4.pth", map_location="cpu")
#     model.eval()
#     return model

# model = load_model()
@st.cache_resource
def load_model(path="temporal_royanet_ssim_model.pth"):
    try:
        obj = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        return None, "not_found"
    except EOFError:
        return None, "eof"
    except Exception as e:
        return None, f"error:{e}"

    # Si se guardó un state_dict (dict de tensores)
    if isinstance(obj, dict):
        # devolver el state_dict para que el usuario sepa que necesita la arquitectura
        return obj, "state_dict"

    # Si se guardó el módulo completo
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj, "module"

    return None, "unknown"

# intentar cargar
model_obj, model_status = load_model()

# Si no hay modelo válido, permitir subirlo desde la UI o usar un fallback dummy
if model_obj is None:
    
    if model_status == "not_found":
        st.warning("No se encontró 'temporal_royanet_ssim_model.pth' en el directorio del proyecto.")
    elif model_status == "eof":
        st.warning("El archivo 'temporal_royanet_ssim_model.pth' parece estar corrupto (EOF). Sube otro archivo válido.")
    else:
        st.warning(f"No se pudo cargar el modelo: {model_status}")

    uploaded_model = st.file_uploader("Sube el archivo del modelo (.pth/.pt)", type=["pth", "pt"])
    if uploaded_model is not None:
        # guardar y reintentar cargar
        path = "modelo_fase4.pth"
        with open(path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model_obj, model_status = load_model(path)

# Si se recibió un state_dict mostramos instrucción porque falta la arquitectura
# ...existing code...
if model_obj is not None and model_status == "state_dict":
    import importlib.util
    import sys
    state_dict = model_obj

    st.info("Se detectó un state_dict (pesos). Necesitamos la clase del modelo para instanciar la arquitectura y cargar los pesos.")

    def load_model_class_from_path(path, candidate_names=("TemporalRoyaNet","MyModel","Model","Net")):
        try:
            spec = importlib.util.spec_from_file_location("model_arch", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name in candidate_names:
                cls = getattr(module, name, None)
                if cls is not None and isinstance(cls, type):
                    return cls
        except Exception:
            return None
        return None

    # Intentar cargar una arquitectura existente en el proyecto llamada model_arch.py
    arch_path = "model_arch.py"
    ModelClass = None
    if os.path.exists(arch_path):
        ModelClass = load_model_class_from_path(arch_path)

    # Si no existe, pedir al usuario que suba un .py con la clase
    if ModelClass is None:
        uploaded_arch = st.file_uploader("Sube un archivo .py con la clase del modelo (p.ej. TemporalRoyaNet)", type=["py"])
        if uploaded_arch is not None:
            with open("model_arch.py", "wb") as f:
                f.write(uploaded_arch.getbuffer())
            ModelClass = load_model_class_from_path("model_arch.py")

    if ModelClass is not None:
        # limpiar posibles prefijos 'module.' de state_dict (p. ej. entrenado con DataParallel)
        new_state = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_k] = v
        try:
            model = ModelClass()
            model.load_state_dict(new_state, strict=False)
            model.eval()
            st.success("Arquitectura cargada y pesos aplicados.")
        except Exception as e:
            st.error(f"Error al instanciar/cargar pesos en la arquitectura: {e}")
            # fallback dummy
            class DummyModel(torch.nn.Module):
                def forward(self, x):
                    return torch.zeros((x.size(0), 1))
            model = DummyModel()
    else:
        st.error("No se encontró la definición de la arquitectura. Opciones:\n"
                 "- Coloca un archivo 'model_arch.py' con la clase (TemporalRoyaNet/MyModel/Model/Net) en el directorio del proyecto.\n"
                 "- O sube el archivo .py usando el uploader.")
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros((x.size(0), 1))
        model = DummyModel()
# ...existing code...
elif model_obj is not None and model_status == "module":
    model = model_obj
else:
    # fallback: dummy model para que la app cargue y puedas probar la interfaz
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros((x.size(0), 1))

    model = DummyModel()

# ----------------------------
#   Transformación de imágenes
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def analyze_images(img1, img2):
    """Retorna SSIM y predicciones usando el modelo."""
    
    t1 = transform(img1).unsqueeze(0)
    t2 = transform(img2).unsqueeze(0)

    with torch.no_grad():
        # PASO 1: PASAR IMÁGENES POR EL MODELO
        out1 = model(t1)
        out2 = model(t2)
        # PASO 2: Calcular SSIM entre imágenes
        ssim_value = ssim(t1, t2).item()
        diff = 1 - ssim_value

    return out1, out2, diff


# ================================
#   INTERFAZ DE USUARIO STREAMLIT
# ================================

st.set_page_config(page_title="TemporalRoyaNet - Comparador", layout="centered")

st.title("🟢 Comparación Temporal – Modelo Fase 4")
st.write("Sube **dos imágenes de hojas** para comparar su diferencia (1 − SSIM) y ver la salida del modelo.")

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
        st.image(img1, caption="Imagen Semana X")
    with colB:
        st.image(img2, caption="Imagen Semana Y")

    st.markdown("---")

    if st.button("🔍 Analizar Variación"):
        out1, out2, diff_value = analyze_images(img1, img2)

        st.subheader("📌 Resultado de la comparación")
        st.metric("Variación entre Semana X - Semana Y", f"{diff_value:.4f}")

        st.markdown("### 🔬 Salida del Modelo (opcional)")
        st.write("Predicción Imagen X:", out1)
        st.write("Predicción Imagen Y:", out2)
else:
    st.info("Carga dos imágenes para comenzar.")
