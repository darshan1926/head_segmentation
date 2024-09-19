import streamlit as st
import numpy as np
from PIL import Image
from app.face_segmentation import FaceSegmentation

# Initialize the segmentation class
face_segmenter = FaceSegmentation()

# Title
st.title("AI Face Segmentation")

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a numpy array
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Validate face count
    detection, error_message = face_segmenter.validate_face_count(image)
    
    if error_message:
        st.error(error_message)
    else:
        # Segment the face
        segmented_face = face_segmenter.segment_face(image, detection)
        
        # Display the segmented face
        st.image(segmented_face, caption="Segmented Face with Transparent Background", use_column_width=True)
        
        # Provide download button
        result = Image.fromarray(segmented_face.astype(np.uint8))
        st.download_button("Download Segmented Face", result, "segmented_face.png")
