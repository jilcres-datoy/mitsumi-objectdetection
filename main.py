import cv2
import streamlit as st

import torch

from yolov5.models.experimental import attempt_load

model_path = 'best.pt'  # replace with your model path
device = 'cpu'  # or 'cuda:0' if you have a GPU

#model = torch.load('./yolov5', 'custom', path='best.pt', force_reload=True)
model = attempt_load(model_path, device=device) 



# Initialize the video capture device
cap = cv2.VideoCapture(0)  # 0 is the default camera index

# Set the maximum width for displaying the video stream
MAX_WIDTH = 800

# Define the Streamlit app layout
st.set_page_config(page_title='Object Detection with YOLOv5', page_icon=':eyeglasses:')
st.title('Object Detection with YOLOv5')

# Loop over the frames in the video stream
while True:
    # Capture the frame from the video stream
    ret, frame = cap.read()

    # Process the frame using the YOLOv5 model
    results = model(frame)

    # Resize the frame to fit the Streamlit app window
    height, width, _ = frame.shape
    scale = min(1, MAX_WIDTH / width)
    display_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    # Draw the YOLOv5 predictions on the frame
    for pred in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = pred
        x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the video stream and the YOLOv5 predictions in the Streamlit app
    st.image(display_frame, caption='Video Stream')

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
