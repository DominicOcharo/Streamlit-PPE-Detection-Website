import PIL
import streamlit as st
from ultralytics import YOLO
import cv2
import time

# Replace the relative path to your weight file
model_path = r'C:\Users\Dell Latitude E6410\Documents\python_projects\gear_results\best.pt'

# Setting page layout
st.set_page_config(
    page_title="PPE Detection",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.markdown("# Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("PPE DETECTION")
st.subheader('Upload a photo and click the button below to analyze it.')

# Creating two columns on the main page
col1, col2 = st.columns([1, 2])

# Initialize requirements list to store detected objects
requirements = []
model = YOLO(model_path)
# Function to perform real-time detection
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Open webcam
    model = YOLO(model_path)    # Load YOLO model

    while cap.isOpened():
        ret, frame = cap.read()  # Read frame from webcam
        if ret:
            res = model.predict(frame, conf=confidence)  # Perform object detection
            boxes = res[0].boxes   # Get bounding boxes
            for box in boxes:
                class_id = res[0].names[box.cls[0].item()]  # Get class name
                requirements.append(class_id)  # Append class name to requirements
                cords = box.xyxy[0].tolist()  # Get coordinates
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)  # Get confidence score

                # Draw bounding box with class label on the frame
                label = f"{class_id} {conf:.2f}"
                cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), (255, 0, 0), 2)
                cv2.putText(frame, label, (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            st.image(frame, caption='Real-time Detection', use_column_width=True)  # Display real-time detection frame

            # Check if "Person" is detected in requirements
            if "Person" in requirements:
                unique_requirements = set(requirements)
                prohibited_items = {item for item in unique_requirements if item.lower().startswith("no")}
                accepted_items = unique_requirements - prohibited_items

                prohibited_percentage = (len(prohibited_items) / 5) * 100
                accepted_percentage = (len(accepted_items) / 5) * 100

                st.subheader("Detection Results:")
                st.write("Prohibited Items:", prohibited_items)
                st.write("Accepted Items:", accepted_items)
                st.write("Prohibited Percentage:", prohibited_percentage)
                st.write("Accepted Percentage:", accepted_percentage)

                if prohibited_percentage >= 50 or accepted_percentage <= 80:
                    st.error("Access Denied!")
                else:
                    st.success("Access Granted!")
                    st.write("Detected Objects:", requirements)
                    st.write("Number of Detected Objects:", len(requirements))
            else:
                st.warning("No Person Detected")
                st.write("Detected Objects:", requirements)

            if st.button('Stop Detection', key=f'stop_detection_{time.time()}'):  # Button to stop real-time detection
                break

        time.sleep(0.1)  # Add a delay of 100 milliseconds

    cap.release()  # Release webcam
    cv2.destroyAllWindows()  # Close OpenCV windows

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    class_id = res[0].names[box.cls[0].item()]
                    requirements.append(class_id)
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    conf = round(box.conf[0].item(), 2)
                    st.write("Object type:", class_id)
                    st.write("Coordinates:", cords)
                    st.write("Probability:", conf)
                    st.write("---")
        except Exception as ex:
            st.write("No image is uploaded yet!")

# Check if "Person" is detected in requirements
if "Person" in requirements:
    unique_requirements = set(requirements)
    prohibited_items = {item for item in unique_requirements if item.lower().startswith("no")}
    accepted_items = unique_requirements - prohibited_items

    prohibited_percentage = (len(prohibited_items) / 5) * 100
    accepted_percentage = (len(accepted_items) / 5) * 100

    st.write("Prohibited Items:", prohibited_items)
    st.write("Accepted Items:", accepted_items)
    st.write("Prohibited Percentage:", prohibited_percentage)
    st.write("Accepted Percentage:", accepted_percentage)

    if prohibited_percentage > 50 or accepted_percentage < 80:
        st.error("Access Denied!")
    else:
        st.success("Access Granted!")
        st.write("Detected Objects:", requirements)
        st.write("Number of Detected Objects:", len(requirements))
else:
    st.warning("No Person Detected")
    st.write("Detected Objects:", requirements)


if st.sidebar.button('Real-time Detection', key='real_time_detection'):
    real_time_detection()

