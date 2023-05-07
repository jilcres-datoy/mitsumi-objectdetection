import streamlit as st
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from video_inference import run_video_inference

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='local', force_reload=True)

st.title("YOLOv5 Streamlit App")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if uploaded_file.type == "bmp":
        image = image.convert("RGB")
        image.save("image.jpg", "JPEG")
        image = Image.open("image.jpg")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize the image
    resize = transforms.Resize([640, 640])
    image = resize(image)

    # Make predictions
    results = model(image)

    results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    results.xyxy[0]  # im predictions (tensor)
    results.pandas().xyxy[0]  # im predictions (pandas)

    # Render the detections on the image
    result_image = results.render()[0]

    # Convert the numpy array to a PIL Image
    result_image = Image.fromarray(result_image)

    # Display the result image with bounding boxes and confidence percentage
    st.image(result_image, caption='Result Image', use_column_width=True)


#video inference
if st.button('Start Video Inference', key='start_button'):
    run_video_inference(model)

    