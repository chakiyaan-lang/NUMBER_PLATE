import os

# YOLO config directory fix
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Streamlit Page Title
st.title("Number Plate Detection App")
st.write("Image upload karein aur YOLO model number plate detect kar ke dega.")


# Model Load Karein
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

# Image Upload Widget
uploaded_file = st.file_uploader(
    "Koi image select karein...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Image Display
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Number Plate"):
        with st.spinner("Processing..."):
            # Prediction
            results = model.predict(image)

            # Bounding box wali image hasil karein
            res_plotted = results[0].plot()

            # OpenCV format (BGR) ko RGB mein convert karein
            res_image = res_plotted[:, :, ::-1]

            # Result Show Karein
            st.success("Detection Complete!")
            st.image(
                res_image, caption="Detected Result", use_container_width=True
            )
