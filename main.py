import streamlit as st
import numpy as np
from PIL import Image
from app.face_segmentation import FaceSegmentation
import io

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
        st.error(error_message)  # Display the error message if there is one
    else:
        # Segment the face
        segmented_face = face_segmenter.segment_face(image)

        if segmented_face is None:
            st.error("No face detected in the image.")
        else:
            # Display the segmented face
            st.image(segmented_face, caption="Segmented Face with Transparent Background", use_column_width=True)

            # Provide download button
            result = Image.fromarray(segmented_face.astype(np.uint8))
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            byte_data = buf.getvalue()
            st.download_button("Download Segmented Face", byte_data, "segmented_face.png")
