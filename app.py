import os
import tempfile

# YOLO config path fix
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(
    page_title="AI Number Plate Detector",
    page_icon="🚗",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    .header-container { text-align: center; padding: 1.5rem 0; }
    .header-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed #4facfe; border-radius: 12px;
        padding: 1rem; background-color: #161b22;
    }
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white; border: none; padding: 0.75rem;
        font-size: 1.1rem; font-weight: 600; border-radius: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🚗 AI License Plate Detector</h1>
        <p style="color: #a0aec0;">Upload Image or Video for Automatic Detection</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Model Loading with Cache
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

# Select File Type Mode
option = st.radio(
    "Choose input type:",
    ["📷 Image Detection", "🎥 Video Detection"],
    horizontal=True,
)

st.markdown("---")

# ----------------- IMAGE MODE -----------------
if option == "📷 Image Detection":
    uploaded_image = st.file_uploader(
        "Drag & Drop Image Here",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Result")
            if st.button("🔍 Detect Number Plate"):
                with st.spinner("Processing image..."):
                    results = model.predict(image)
                    res_plotted = results[0].plot()
                    res_image = res_plotted[:, :, ::-1]  # BGR to RGB
                    st.image(
                        res_image,
                        caption="Detected Result",
                        use_container_width=True,
                    )
                    st.success("✅ Detection Complete!")

# ----------------- VIDEO MODE -----------------
else:
    uploaded_video = st.file_uploader(
        "Drag & Drop Video Here",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📹 Original Video")
            st.video(tfile.name)

        with col2:
            st.subheader("🎯 Output Video")

            if st.button("⚡ Process Video"):
                with st.spinner("Detecting plates frame-by-frame..."):
                    cap = cv2.VideoCapture(tfile.name)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    output_path = tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp4"
                    ).name
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        output_path, fourcc, fps, (width, height)
                    )

                    st_frame = st.empty()

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # Process frame with YOLO
                        results = model.predict(frame, conf=0.3)
                        res_plotted = results[0].plot()

                        # Write to output file
                        out.write(res_plotted)

                        # Live Streamlit Preview (BGR to RGB)
                        st_frame.image(
                            res_plotted[:, :, ::-1],
                            channels="RGB",
                            use_container_width=True,
                        )

                    cap.release()
                    out.release()

                    st.success("✅ Video Processing Complete!")
