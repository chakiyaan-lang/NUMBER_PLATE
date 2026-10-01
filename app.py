from ultralytics import YOLO
import gradio as gr 


model = YOLO("best (1).pt")

def pred_image(image):
    img = model.predict(image)
    return img[0].plot()


app= gr.Interface(fn = pred_image, inputs = 'image', outputs = "image" )
app.launch()

# download libraries ---> pip install -r requirements.txt
# break the terminal   ---- ctrl +c
# run the command that opeen broeswe
#  python app.py