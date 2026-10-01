import os

# YOLO config path fix
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from PIL import Image
import streamlit as st
from ultralytics import YOLO

# Page Configuration (Wide Layout & Custom Title)
st.set_page_config(
    page_title="AI Number Plate Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Modern UI & Styling
st.markdown(
    """
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Header Container */
    .header-container {
        text-align: center;
        padding: 2rem 0rem 1.5rem 0rem;
    }
    .header-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
    }

    /* Style File Uploader (Drag & Drop Zone) */
    [data-testid="stFileUploader"] {
        border: 2px dashed #4facfe;
        border-radius: 12px;
        padding: 1.5rem;
        background-color: #161b22;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00f2fe;
        background-color: #1f242d;
    }

    /* Custom Button Style */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.6);
    }
    </style>
""",
    unsafe_allow_html=True,
)

# App Header
st.markdown(
    """
    <div class="header-container">
        <h1 class="header-title">🚗 AI Number Plate Detector</h1>
        <p class="header-subtitle">Upload or Drag & Drop any vehicle image to detect license plates instantly</p>
    </div>
""",
    unsafe_allow_html=True,
)


# Model Loading with Cache
@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

# Centered Section for Upload
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Drag & Drop your image here",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP",
    )

# Predictions & Result Display Layout
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown("---")

    # Side-by-side comparison layout
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_container_width=True)

    with img_col2:
        st.subheader("🎯 Detection Result")

        # Process Button & Output Container
        if st.button("🔍 Detect Number Plate"):
            with st.spinner("Analyzing image with YOLO..."):
                results = model.predict(image)

                # Plot results
                res_plotted = results[0].plot()
                res_image = res_plotted[:, :, ::-1]  # BGR to RGB

                st.image(
                    res_image,
                    caption="Number Plate Detected",
                    use_container_width=True,
                )
                st.success("✅ Detection completed successfully!")
