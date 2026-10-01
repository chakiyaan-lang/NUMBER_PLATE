import os

# Set YOLO config path first
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

import gradio as gr
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")


def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


# Create Gradio Interface
app = gr.Interface(fn=pred_image, inputs="image", outputs="image")

# Launch App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.launch(server_name="0.0.0.0", server_port=port, share=True)
