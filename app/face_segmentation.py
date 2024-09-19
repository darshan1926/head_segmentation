import cv2
import numpy as np
import logging
import mediapipe as mp
from PIL import Image
from config import Config

# Set up logger
logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class FaceSegmentation:
    def __init__(self):
        # Initialize Mediapipe models
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=Config.MODEL_SELECTION, 
                                                                        min_detection_confidence=Config.CONFIDENCE_THRESHOLD)
        logging.info("Initialized FaceSegmentation class with model selection %s and confidence threshold %s", 
                     Config.MODEL_SELECTION, Config.CONFIDENCE_THRESHOLD)

    def validate_face_count(self, image):
        """Validates that the image contains exactly one face."""
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.detections:
            logging.warning("No face detected in the image.")
            return None, "No face detected. Please upload an image with exactly one face."
        elif len(results.detections) > 1:
            logging.warning("Multiple faces detected in the image.")
            return None, "Multiple faces detected. Please upload an image containing only one face."
        
        logging.info("Successfully detected one face.")
        return results.detections[0], None

    def segment_face(self, image, detection):
        """Segments the face from the image and returns it with transparent background."""
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        
        # Crop the face
        face = image[y:y+h, x:x+w]
        logging.info("Cropped the face from the image.")
        
        # Remove background from the cropped face
        bg_removed = self.remove_background(face)
        return bg_removed

    def remove_background(self, face_img):
        logging.info("Starting background removal...")

        # Check the number of channels in the image
        if len(face_img.shape) == 3:
            if face_img.shape[2] == 4:
                logging.info("Image has 4 channels (RGBA). Converting to BGR...")
                # Convert from RGBA to BGR by dropping the alpha channel
                face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
                logging.info("Converted RGBA to BGR.")
            elif face_img.shape[2] == 1:
                logging.warning("Image is single-channel (grayscale). Converting to BGR...")
                # Convert grayscale to BGR
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                logging.info("Converted grayscale to BGR.")
            elif face_img.shape[2] != 3:
                logging.error(f"Unexpected number of channels: {face_img.shape[2]}")
                raise ValueError(f"Unexpected number of channels: {face_img.shape[2]}")

        else:
            logging.error(f"Invalid image shape: {face_img.shape}")
            raise ValueError(f"Invalid image shape: {face_img.shape}")

        # Initialize mask, background, and foreground models
        mask = np.zeros(face_img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # Define a rectangle for grabCut (this can be adjusted based on the image size)
        rect = (1, 1, face_img.shape[1] - 1, face_img.shape[0] - 1)

        try:
            # Perform grabCut to remove background
            logging.info("Running grabCut for background removal...")
            cv2.grabCut(face_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            logging.info("grabCut completed successfully.")
        except cv2.error as e:
            logging.error(f"OpenCV Error during grabCut: {e}")
            raise

        # Modify the mask to remove background
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        face_img = face_img * mask2[:, :, np.newaxis]

        logging.info("Background removal completed successfully.")

        return face_img