import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.predict import load_trained_model, predict_digit
from utils.noise import add_noise


try:
    RESAMPLING = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLING = Image.LANCZOS


st.set_page_config(
    page_title="Prediction de chiffre",
    page_icon="8",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(251, 191, 36, 0.16), transparent 24%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 28%),
            linear-gradient(180deg, #fffaf0 0%, #f5f9ff 100%);
        color: #12243d;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .hero-card, .panel-card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 22px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(8px);
    }
    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }
    .hero-text {
        color: #36506d;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .section-tag {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #b45309;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .pill {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        margin-right: 0.45rem;
        margin-top: 0.55rem;
        font-size: 0.92rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_trained_model()


def ensure_grayscale(image: Image.Image) -> Image.Image:
    return image.convert("L")


def crop_digit(array: np.ndarray, threshold: float) -> np.ndarray:
    mask = array > threshold
    if not np.any(mask):
        return array

    coords = np.argwhere(mask)
    top, left = coords.min(axis=0)
    bottom, right = coords.max(axis=0) + 1
    return array[top:bottom, left:right]


def square_and_resize(array: np.ndarray, size: int = 28) -> np.ndarray:
    image = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("L", (side, side), color=0)
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    resized = canvas.resize((size, size), RESAMPLING)
    return np.asarray(resized, dtype=np.float32) / 255.0


def prepare_image(
    image: Image.Image,
    invert_mode: str,
    trim_digit: bool,
    threshold: float,
    contrast_factor: float,
    sharpen_factor: float,
    noise_factor: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    grayscale = ensure_grayscale(image)
    enhanced = ImageOps.autocontrast(grayscale)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast_factor)

    if sharpen_factor > 1:
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpen_factor)
        enhanced = enhanced.filter(ImageFilter.DETAIL)

    original_preview = np.asarray(grayscale, dtype=np.float32) / 255.0
    processed = np.asarray(enhanced, dtype=np.float32) / 255.0

    if invert_mode == "Auto":
        if processed.mean() > 0.5:
            processed = 1.0 - processed
    elif invert_mode == "Oui":
        processed = 1.0 - processed

    if trim_digit:
        processed = crop_digit(processed, threshold)

    processed = square_and_resize(processed, 28)

    if noise_factor > 0:
        processed = add_noise(processed.reshape(28, 28, 1), noise_factor=noise_factor).reshape(28, 28)

    metadata = {
        "source_size": f"{grayscale.width}x{grayscale.height}",
        "mean_intensity": round(float(original_preview.mean()), 3),
        "contrast_factor": contrast_factor,
        "sharpen_factor": sharpen_factor,
        "trimmed": trim_digit,
        "noise_factor": noise_factor,
    }
    return original_preview, processed, metadata


def predict_with_report(model, image_array: np.ndarray) -> dict:
    digit, confidence, probabilities = predict_digit(model, image_array)
    ordered = np.argsort(probabilities)[::-1]
    top_predictions = [
        {"digit": int(idx), "probability": round(float(probabilities[idx]) * 100, 2)}
        for idx in ordered[:3]
    ]
    return {
        "digit": digit,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
    }


def probability_dataframe(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "digit": [str(i) for i in range(10)],
            "probability": [round(float(value) * 100, 2) for value in probabilities],
        }
    )


def build_report(prediction: dict, metadata: dict, source_name: str) -> str:
    payload = {
        "source": source_name,
        "predicted_digit": prediction["digit"],
        "confidence_percent": round(float(prediction["confidence"]), 2),
        "top_predictions": prediction["top_predictions"],
        "metadata": metadata,
    }
    return json.dumps(payload, indent=2)


def image_to_bytes(image_array: np.ndarray) -> bytes:
    output = BytesIO()
    image = Image.fromarray((image_array * 255).astype(np.uint8), mode="L")
    image.save(output, format="PNG")
    return output.getvalue()


st.markdown(
    """
    <div class="hero-card">
        <div class="section-tag">Interface de prediction</div>
        <div class="hero-title">Upload une photo brouillee de chiffre</div>
        <p class="hero-text">
            Depose une photo floue ou imparfaite, puis laisse l'application la nettoyer,
            la recentrer et la convertir au format attendu par le modele.
        </p>
        <span class="pill">Upload photo</span>
        <span class="pill">Pretraitement auto</span>
        <span class="pill">Prediction</span>
        <span class="pill">Robustesse au bruit</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

control_col, uploader_col = st.columns([1, 1.2], gap="large")

with control_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Reglages")
    invert_mode = st.selectbox("Couleurs", ["Auto", "Non", "Oui"], index=0)
    trim_digit_enabled = st.toggle("Recadrer autour du chiffre", value=True)
    compare_noise = st.toggle("Afficher le test de robustesse", value=True)

    with st.expander("Reglages avances"):
        threshold = st.slider("Seuil de detection", 0.01, 0.50, 0.12, 0.01)
        contrast_factor = st.slider("Renforcer le contraste", 1.0, 3.0, 1.8, 0.1)
        sharpen_factor = st.slider("Accentuer la nettete", 1.0, 4.0, 2.2, 0.1)
        noise_factor = st.slider("Ajouter du bruit de test", 0.0, 0.8, 0.0, 0.05)

    st.caption("Conseil: cadre surtout un seul chiffre, avec un fond simple si possible.")
    st.markdown("</div>", unsafe_allow_html=True)

with uploader_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Photo a analyser")
    uploaded_file = st.file_uploader(
        "Importer une photo brouillee",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    st.caption("Formats acceptes: PNG, JPG, JPEG, BMP, WEBP")
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Ajoute une photo de chiffre pour lancer l'analyse.")
    st.stop()

try:
    model = get_model()
except ModuleNotFoundError as exc:
    st.error(str(exc))
    st.stop()

image = Image.open(uploaded_file)
original_preview, processed_image, metadata = prepare_image(
    image=image,
    invert_mode=invert_mode,
    trim_digit=trim_digit_enabled,
    threshold=threshold,
    contrast_factor=contrast_factor,
    sharpen_factor=sharpen_factor,
    noise_factor=noise_factor,
)
prediction = predict_with_report(model, processed_image)
probability_df = probability_dataframe(prediction["probabilities"])
report_json = build_report(prediction, metadata, uploaded_file.name)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Chiffre predit", str(prediction["digit"]))
metric_col2.metric("Confiance", f"{prediction['confidence']:.2f}%")
metric_col3.metric("Taille source", metadata["source_size"])

preview_tab, score_tab, export_tab = st.tabs(["Apercu", "Scores", "Export"])

with preview_tab:
    left_col, right_col = st.columns(2, gap="large")
    left_col.image(original_preview, caption="Photo d'origine", clamp=True)
    right_col.image(processed_image, caption="Image envoyee au modele", clamp=True)

with score_tab:
    insight_col, chart_col = st.columns([0.9, 1.1], gap="large")
    with insight_col:
        st.markdown(
            f"""
            <div class="panel-card">
                <strong>Lecture rapide</strong><br/><br/>
                Top 1: <strong>{prediction['top_predictions'][0]['digit']}</strong> avec <strong>{prediction['top_predictions'][0]['probability']:.2f}%</strong><br/>
                Top 2: <strong>{prediction['top_predictions'][1]['digit']}</strong> avec <strong>{prediction['top_predictions'][1]['probability']:.2f}%</strong><br/>
                Top 3: <strong>{prediction['top_predictions'][2]['digit']}</strong> avec <strong>{prediction['top_predictions'][2]['probability']:.2f}%</strong><br/><br/>
                Intensite moyenne: <strong>{metadata['mean_intensity']}</strong><br/>
                Contraste: <strong>{metadata['contrast_factor']}</strong><br/>
                Nettete: <strong>{metadata['sharpen_factor']}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with chart_col:
        st.bar_chart(probability_df.set_index("digit"))
        st.dataframe(probability_df, use_container_width=True, hide_index=True)

with export_tab:
    export_col1, export_col2 = st.columns(2, gap="large")
    with export_col1:
        st.download_button(
            "Telecharger l'image pretraitee",
            data=image_to_bytes(processed_image),
            file_name=f"processed_{Path(uploaded_file.name).stem}.png",
            mime="image/png",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            "Telecharger le rapport JSON",
            data=report_json,
            file_name=f"prediction_{Path(uploaded_file.name).stem}.json",
            mime="application/json",
            use_container_width=True,
        )
    st.code(report_json, language="json")

if compare_noise:
    st.subheader("Robustesse au bruit")
    noise_levels = [0.0, 0.15, 0.30, 0.45]
    robustness_rows = []
    noise_columns = st.columns(len(noise_levels))
    base_image = processed_image.reshape(28, 28, 1)

    for idx, level in enumerate(noise_levels):
        noisy_variant = add_noise(base_image, noise_factor=level).reshape(28, 28)
        noisy_prediction = predict_with_report(model, noisy_variant)
        robustness_rows.append(
            {
                "noise": level,
                "digit": noisy_prediction["digit"],
                "confidence": round(float(noisy_prediction["confidence"]), 2),
            }
        )
        noise_columns[idx].image(noisy_variant, caption=f"Bruit {level:.2f}", clamp=True)
        noise_columns[idx].metric("Pred", str(noisy_prediction["digit"]))
        noise_columns[idx].metric("Conf", f"{noisy_prediction['confidence']:.2f}%")

    st.dataframe(pd.DataFrame(robustness_rows), use_container_width=True, hide_index=True)
