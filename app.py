from ultralytics import YOLO
import gradio as gr 


model = YOLO("best.pt")

def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


app= gr.Interface(fn = pred_image, inputs = 'image', outputs = "image" ),share=True,
app.launch()
app.launch(
    server_name="0.0.0.0",
    server_port=int(__import__("os").environ.get("PORT", 10000)))
import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

from ultralytics import YOLO
# Baki aapka code...
