import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt

def find_defects(master_img, input_img, threshold):
    # Convert to grayscale
    master_gray = cv2.cvtColor(master_img, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and minor variations
    master_gray_blur = cv2.GaussianBlur(master_gray, (5, 5), 0)
    input_gray_blur = cv2.GaussianBlur(input_gray, (5, 5), 0)

    # Image subtraction
    diff_img = cv2.absdiff(master_gray_blur, input_gray_blur)

    # Apply precise thresholding to highlight differences
    _, thresh_img = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)

    # Use morphological operations to enhance the differences
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(thresh_img, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects_present = False

    # Draw bounding boxes around the defects
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Adjust the area threshold for smaller defects
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(input_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            defects_present = True

    return input_img, defects_present, len(contours)

def record_defect_data(master_image_path, input_image_path, defects_present, num_defects, threshold, defective_image_path):
    # Load existing data
    try:
        df = pd.read_excel("defect_detection_results.xlsx")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Timestamp", "Master Image", "Input Image", "Defects Present", "Number of Defects", "Threshold", "Defective Image"])

    # Create new record
    new_record = {
        "Timestamp": datetime.datetime.now(),
        "Master Image": master_image_path,
        "Input Image": input_image_path,
        "Defects Present": defects_present,
        "Number of Defects": num_defects,
        "Threshold": threshold,
        "Defective Image": defective_image_path
    }

    # Append new record to dataframe using pd.concat
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    # Save to Excel file
    df.to_excel("defect_detection_results.xlsx", index=False)

def visualize_data():
    # Load data
    try:
        df = pd.read_excel("defect_detection_results.xlsx")
    except FileNotFoundError:
        st.error("No records found.")
        return

    # Plot data
    fig, ax = plt.subplots()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = df["Timestamp"].dt.date
    defect_counts = df.groupby("Date").size()

    defect_counts.plot(kind="bar", ax=ax)
    ax.set_title("Number of Defective Circuits per Day")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Defects")
    st.pyplot(fig)

# Streamlit application
st.title("Defect Detection Application")

# Option to upload or take photos
option = st.selectbox(
    "Select image input method",
    ("Upload Images", "Use Camera")
)

# Initialize variables
master_img = None
input_img = None

# Threshold slider
threshold = st.slider("Set threshold value for defect detection", 0, 255, 5)

# Upload images or use camera
if option == "Upload Images":
    uploaded_master = st.file_uploader("Choose the master image...", type=["jpg", "jpeg", "png"])
    uploaded_input = st.file_uploader("Choose the input image...", type=["jpg", "jpeg", "png"])

    if uploaded_master is not None and uploaded_input is not None:
        master_img = Image.open(uploaded_master)
        input_img = Image.open(uploaded_input)

        master_image_path = uploaded_master.name
        input_image_path = uploaded_input.name

elif option == "Use Camera":
    master_img = st.camera_input("Take a photo of the master image")
    input_img = st.camera_input("Take a photo of the input image")

    if master_img is not None and input_img is not None:
        master_img = Image.open(master_img)
        input_img = Image.open(input_img)

        master_image_path = "Camera master image"
        input_image_path = "Camera input image"

# Process images if both are provided
if master_img is not None and input_img is not None:
    master_img = np.array(master_img)
    input_img = np.array(input_img)

    if master_img.shape == input_img.shape:
        # Detect defects
        result_img, defects_present, num_defects = find_defects(master_img, input_img, threshold)

        # Convert result image to display in Streamlit
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)

        # Save defective image
        if not os.path.exists("defective_images"):
            os.makedirs("defective_images")
        defective_image_path = f"defective_images/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_defective.png"
        result_pil.save(defective_image_path)

        # Display images
        st.image(result_pil, caption='Output Image with Defects Marked', use_column_width=True)

        # Display message if no defects are found
        if not defects_present:
            st.success("No defects present.")
        else:
            st.warning(f"{num_defects} defects found.")

        # Record data to Excel
        record_defect_data(master_image_path, input_image_path, defects_present, num_defects, threshold, defective_image_path)
    else:
        st.error("The master and input images do not have the same dimensions.")
else:
    st.warning("Please provide both the master image and the input image.")

# Button to view records
if st.button("View Records"):
    visualize_data()

    # Display records
    try:
        df = pd.read_excel("defect_detection_results.xlsx")
        st.dataframe(df)
    except FileNotFoundError:
        st.error("No records found.")
