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
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=Config.MODEL_SELECTION, 
                                                                    min_detection_confidence=Config.CONFIDENCE_THRESHOLD)

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