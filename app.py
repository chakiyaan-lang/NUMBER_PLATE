import os
import tempfile

os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="AI License Plate Detector",
    page_icon="🚗",
    layout="wide",
)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

st.title("🚗 AI License Plate Detector")

option = st.radio(
    "Select Mode:",
    ["📷 Image Detection", "🎥 Video Detection"],
    horizontal=True,
)

# ---------------- IMAGE MODE ----------------
if option == "📷 Image Detection":
    uploaded_image = st.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            if st.button("Detect Number Plate"):
                results = model.predict(image, conf=0.25)
                res_plotted = results[0].plot()
                st.image(
                    res_plotted[:, :, ::-1],
                    caption="Detection Result",
                    use_container_width=True,
                )

# ---------------- VIDEO MODE (FAST & WORKING) ----------------
else:
    uploaded_video = st.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        st.subheader("Live Detection Output")
        st_frame = st.empty()  # Live preview window
        progress_bar = st.progress(0)

        if st.button("⚡ Start Detection"):
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame-by-frame prediction
                results = model.predict(frame, conf=0.25)
                res_plotted = results[0].plot()

                # Streamlit screen par LIVE frame show karein
                st_frame.image(
                    res_plotted[:, :, ::-1],
                    caption="Processing Video...",
                    use_container_width=True,
                )

                # Progress update
                frame_count += 1
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            st.success("✅ Detection Finished!")
