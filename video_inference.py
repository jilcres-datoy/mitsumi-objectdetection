import cv2
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

def run_video_inference(model):
    # Video capture
    cap = cv2.VideoCapture(0) # 0 for webcam or path to video file

    frameST = st.empty()

    stop_button = st.button('Stop Video Inference', key='stop_button')

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Resize the image
        resize = transforms.Resize([640, 640])
        image = resize(image)

        # Make predictions
        results = model(image)

        # Render the detections on the image
        result_image = results.render()[0]

        # Convert the numpy array to a PIL Image
        result_image = Image.fromarray(result_image)

        # Display the result image with bounding boxes and confidence percentage
        frameST.image(result_image, caption='Result Image', use_column_width=True)

    cap.release()
    
    # Display the detected classes and their confidence scores
    for class_name, confidence in zip(results.xyxyn[0][:, -1], results.xyxyn[0][:, -2]):
        st.write(f"{class_name}: {confidence:.2f}")