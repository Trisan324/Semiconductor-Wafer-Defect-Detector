from pathlib import Path

import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import CompactCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "compact_cnn_best.pt"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_wafer_dataset.npz"


@st.cache_resource
def load_model(num_classes):
    model = CompactCNN(num_classes=num_classes)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_data
def load_test_data():
    data = np.load(DATA_PATH, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"].argmax(axis=1)
    class_names = data["class_names"].tolist()
    return X_test, y_test, class_names


def predict(model, wafer_map):
    x = torch.tensor(wafer_map, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    predicted_idx = int(probs.argmax().item())
    return predicted_idx, probs.numpy()


st.set_page_config(page_title="Wafer Defect Classifier", layout="wide")

# hides Streamlit's default header/menu/footer and trims padding so
# everything fits on one screen without scrolling
st.markdown(
    """
    <style>
    #MainMenu, header, footer { visibility: hidden; }
    .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    h1 { margin-top: 0rem; font-size: 1.6rem; }
    [data-testid="stVerticalBlock"] { gap: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Semiconductor Wafer Defect Classifier")

X_test, y_test, class_names = load_test_data()
model = load_model(num_classes=len(class_names))

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = 0

left, right = st.columns(2)

with left:
    nav_prev, nav_random, nav_next = st.columns(3)
    with nav_prev:
        if st.button("<", use_container_width=True):
            st.session_state.sample_idx = (st.session_state.sample_idx - 1) % len(X_test)
    with nav_random:
        if st.button("Random", use_container_width=True):
            st.session_state.sample_idx = int(np.random.randint(0, len(X_test)))
    with nav_next:
        if st.button(">", use_container_width=True):
            st.session_state.sample_idx = (st.session_state.sample_idx + 1) % len(X_test)

    idx = st.session_state.sample_idx
    st.caption(f"Test sample {idx} of {len(X_test) - 1}")

    wafer_map = X_test[idx]
    true_label = class_names[y_test[idx]]

    # one-hot channels (background/normal/defective) collapse back to a single
    # categorical map for display, same as the original raw wafer map values
    display_map = wafer_map.argmax(axis=-1)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(display_map, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig, use_container_width=False)

with right:
    predicted_idx, probs = predict(model, wafer_map)
    predicted_label = class_names[predicted_idx]
    confidence = probs[predicted_idx] * 100

    st.subheader(f"Predicted: {predicted_label} ({confidence:.1f}% confidence)")
    st.write(f"True label: {true_label}")
    st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})