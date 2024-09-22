import cv2
import numpy as np
import logging
import mediapipe as mp
from PIL import Image
from config import Config
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights
# # import face_recognition

# # Set up logger
# logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class FaceSegmentation:
    
    def __init__(self):
        # Load pre-trained DeepLabV3 model for human segmentation using the updated 'weights' parameter
        self.weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = models.segmentation.deeplabv3_resnet101(weights=self.weights).eval()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
#     def __init__(self):
#         # Initialize Mediapipe models
#         # Load pre-trained model for head detection (for simplicity, using a Haar Cascade for face detection as an example)
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=Config.MODEL_SELECTION, 
                                                                    min_detection_confidence=Config.CONFIDENCE_THRESHOLD)
#         # Load pre-trained DeepLabV3 model for human segmentation using the updated 'weights' parameter
#         weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
#         model = models.segmentation.deeplabv3_resnet101(weights=weights).eval()
#         logging.info("Initialized FaceSegmentation class with model selection %s and confidence threshold %s",
#                     Config.MODEL_SELECTION, Config.CONFIDENCE_THRESHOLD)

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

#     def segment_face(self, image, detection):
#         # Convert the input image to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the image
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#         # Initialize an empty variable for head_region
#         head_region = None

#         # Assuming the head region to be above and around the detected face
#         head_boxes = []
#         for (x, y, w, h) in faces:
#             # Define the head region (extend some pixels above the face detection)
#             head_top_y = max(0, y - int(h * 1.2))  # Extend the head region above the face
#             head_bottom_y = y + int(h * 0.2)  # Optional: Include a bit of area below the face
#             head_boxes.append((x, head_top_y, w, head_bottom_y - head_top_y))

#         # Process the head region by excluding the face region
#         if head_boxes:  # Ensure there are detected head regions
#             for (x, y, w, h) in head_boxes:
#                 # Extract head region
#                 head_region = image[y:y+h, x:x+w]

#                 # Blur or remove the face from the head region (for privacy or other reasons)
#                 for (fx, fy, fw, fh) in faces:
#                     # Only process faces that overlap with the head region
#                     if x < fx < x+w and y < fy < y+h:
#                         # Blur the face inside the head region
#                         face_region = head_region[fy-y:fy-y+fh, fx-x:fx-x+fw]
#                         blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
#                         head_region[fy-y:fy-y+fh, fx-x:fx-x+fw] = blurred_face

#         if head_region is None:
#             logging.error("No head region was detected or processed.")
#             return None

#         return head_region
#         # """Segments the face from the image and returns it with transparent background."""
#         # bboxC = detection.location_data.relative_bounding_box
#         # ih, iw, _ = image.shape
#         # x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        
#         # # Crop the face
#         # face = image[y:y+h, x:x+w]
#         # logging.info("Cropped the face from the image.")
        
#         # # Remove background from the cropped face
#         # # bg_removed = self.remove_background(face)
#         # return face

#     def remove_background(self, face_img):
#         logging.info("Starting background removal...")

#         # Check the number of channels in the image
#         if len(face_img.shape) == 3:
#             if face_img.shape[2] == 4:
#                 logging.info("Image has 4 channels (RGBA). Converting to BGR...")
#                 # Convert from RGBA to BGR by dropping the alpha channel
#                 face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2BGR)
#                 logging.info("Converted RGBA to BGR.")
#             elif face_img.shape[2] == 1:
#                 logging.warning("Image is single-channel (grayscale). Converting to BGR...")
#                 # Convert grayscale to BGR
#                 face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
#                 logging.info("Converted grayscale to BGR.")
#             elif face_img.shape[2] != 3:
#                 logging.error(f"Unexpected number of channels: {face_img.shape[2]}")
#                 raise ValueError(f"Unexpected number of channels: {face_img.shape[2]}")

#         else:
#             logging.error(f"Invalid image shape: {face_img.shape}")
#             raise ValueError(f"Invalid image shape: {face_img.shape}")

#         # Initialize mask, background, and foreground models
#         mask = np.zeros(face_img.shape[:2], np.uint8)
#         bgdModel = np.zeros((1, 65), np.float64)
#         fgdModel = np.zeros((1, 65), np.float64)

#         # Define a rectangle for grabCut (this can be adjusted based on the image size)
#         rect = (1, 1, face_img.shape[1] - 1, face_img.shape[0] - 1)

#         try:
#             # Perform grabCut to remove background
#             logging.info("Running grabCut for background removal...")
#             cv2.grabCut(face_img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
#             # image = face_recognition.load_image_file(face_img)
#             # face_locations = face_recognition.face_locations(image)
#             logging.info("grabCut completed successfully.")
#         except cv2.error as e:
#             logging.error(f"OpenCV Error during grabCut: {e}")
#             raise

#         # Modify the mask to remove background
#         mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
#         face_img = face_img * mask2[:, :, np.newaxis]

#         logging.info("Background removal completed successfully.")

#         return face_img
    def preprocess_image(self, image):
        input_image = Image.fromarray(image).convert("RGB")
        input_tensor = self.preprocess(input_image).unsqueeze(0)  # Add batch dimension
        return input_tensor, input_image
    
    def get_segmentation_mask(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        return output_predictions
    
    def extract_head(self, image, mask):
        PERSON_CLASS = 15
        person_mask = (mask == PERSON_CLASS).astype(np.uint8)

        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        person_crop = image[y:y+h, x:x+w]
        person_mask_crop = person_mask[y:y+h, x:x+w]

        head_height = int(h * 0.7)
        head_mask = person_mask_crop.copy()
        head_mask[head_height:, :] = 0

        kernel = np.ones((5, 5), np.uint8)
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_CLOSE, kernel)
        head_mask = cv2.morphologyEx(head_mask, cv2.MORPH_OPEN, kernel)

        head_cutout = cv2.bitwise_and(person_crop, person_crop, mask=head_mask)
        return head_cutout

    def segment_face(self, image):
        input_tensor, pil_image = self.preprocess_image(image)
        segmentation_mask = self.get_segmentation_mask(input_tensor)
        head_cutout = self.extract_head(image, segmentation_mask)
        return head_cutout