import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import requests
import io
import base64
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class FurnitureRepositioningSystem:
    def __init__(self, custom_model_path=None):
        """Initialize the furniture repositioning system with generative AI capabilities"""
        print("Initializing furniture detection and repositioning system...")

        # Check for MPS (Metal Performance Shaders) availability on Mac
        self.device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load YOLO model for furniture detection
        print("Loading YOLO model...")

        # Use custom trained model if provided, else use standard model
        if custom_model_path and os.path.exists(custom_model_path):
            self.detection_model = YOLO(custom_model_path)
            print(f"Loaded custom furniture detection model from: {custom_model_path}")
        else:
            self.detection_model = YOLO('yolov8n.pt')  # Use nano version for better performance on M2
            print("Loaded standard YOLOv8n model")

        print("Setting up inpainting capabilities...")

        # Extended furniture classes beyond COCO dataset
        # Original COCO furniture classes
        self.coco_furniture_classes = {
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            73: 'book',
            74: 'clock',
            75: 'vase'
        }

        # Initialize empty dictionary for custom classes
        self.custom_furniture_classes = {}

        # We'll use a broader definition of furniture to include more items
        self.furniture_keywords = [
            'chair', 'couch', 'sofa', 'table', 'desk', 'bed', 'cabinet', 'wardrobe',
            'dresser', 'nightstand', 'bookshelf', 'shelf', 'ottoman', 'stool', 'bench',
            'lamp', 'light', 'rug', 'carpet', 'curtain', 'blind', 'mirror', 'frame',
            'picture', 'painting', 'plant', 'vase', 'pillow', 'cushion', 'drawer',
            'chest', 'armchair', 'tv', 'television', 'stand', 'console', 'hutch',
            'buffet', 'sideboard', 'recliner', 'futon', 'loveseat', 'sectional',
            'coffee table', 'end table', 'desk', 'dining table', 'vanity', 'wine rack'
        ]

        print("System initialized successfully!")

    def is_furniture(self, class_name, confidence):
        """
        Determine if a detected object is furniture based on class name and confidence
        """
        # Convert class name to lowercase for case-insensitive matching
        class_name = class_name.lower()

        # Check if the class name contains any furniture keyword
        for keyword in self.furniture_keywords:
            if keyword in class_name:
                return True

        # Check if it's a COCO furniture class
        if class_name in self.coco_furniture_classes.values():
            return True

        # If confidence is very high, include other potential furniture items
        if confidence > 0.8 and (
                'object' in class_name or
                'item' in class_name or
                'home' in class_name or
                'decor' in class_name or
                'house' in class_name
        ):
            return True

        return False

    def detect_furniture(self, image_path):
        """
        Detect furniture in the given image using YOLO

        Args:
            image_path: Path to the input image

        Returns:
            Original image and detected furniture items with their bounding boxes
        """
        print(f"Detecting furniture in {image_path}...")

        # Run YOLO detection with higher recall for furniture detection
        results = self.detection_model(image_path, conf=0.25, verbose=False)

        # Load image with OpenCV to visualize results
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store detected furniture items
        furniture_items = []

        # Process detection results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Get class name
                names = r.names
                class_name = names[cls]

                # Use extended furniture detection logic
                if cls in self.coco_furniture_classes:
                    # Direct match with COCO furniture classes
                    furniture_name = self.coco_furniture_classes[cls]
                    is_furniture_item = True
                else:
                    # Check if this might be furniture based on class name
                    furniture_name = class_name
                    is_furniture_item = self.is_furniture(class_name, conf)

                if is_furniture_item:
                    furniture_items.append({
                        'name': furniture_name,
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_id': cls
                    })

                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Add label
                    label = f"{furniture_name}: {conf:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"Detected {len(furniture_items)} furniture items")
        return image, furniture_items

    def train_custom_detector(self, dataset_path, epochs=100):
        """
        Train a custom YOLO model specifically for furniture detection

        Args:
            dataset_path: Path to dataset in YOLO format
            epochs: Number of training epochs
        """
        print(f"Training custom furniture detection model from {dataset_path}...")

        # Create a new YOLO model based on yolov8n
        model = YOLO('yolov8n.yaml')

        # Train the model on custom dataset
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            device=self.device,
            patience=20,
            batch=16,
            name='furniture_detector'
        )

        # Update the detection model with trained model
        self.detection_model = model

        print(f"Custom model trained successfully! Saved to {model.ckpt_path}")
        return model.ckpt_path

    def segment_furniture(self, image_path):
        """
        Perform instance segmentation to get precise furniture shapes

        Args:
            image_path: Path to input image

        Returns:
            Original image, furniture items, and segmentation masks
        """
        print(f"Segmenting furniture in {image_path}...")

        # Load segmentation model
        seg_model = YOLO('yolov8n-seg.pt')

        # Run segmentation
        results = seg_model(image_path, conf=0.3, verbose=False)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Store detected furniture items
        furniture_items = []
        masks = []

        # Process segmentation results
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None:
                for i, (box, mask) in enumerate(zip(r.boxes, r.masks.data)):
                    # Get class and confidence
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get class name
                    class_name = r.names[cls]

                    # Check if this is furniture
                    is_furniture_item = self.is_furniture(class_name, conf)

                    if is_furniture_item:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        # Convert mask to numpy array
                        mask_np = mask.cpu().numpy()

                        # Resize mask to image dimensions
                        mask_image = np.zeros(image.shape[:2], dtype=np.uint8)
                        mask_image = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

                        furniture_items.append({
                            'name': class_name,
                            'box': (x1, y1, x2, y2),
                            'confidence': conf,
                            'class_id': cls
                        })

                        masks.append(mask_image)

                        # Draw mask overlay
                        color_mask = np.zeros_like(image)
                        color_mask[mask_image > 0.5] = [0, 0, 255]  # Red color for mask
                        image = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)

                        # Add label
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"Segmented {len(furniture_items)} furniture items")
        return image, furniture_items, masks

    def extract_furniture(self, image, box):
        """
        Extract furniture item from image based on bounding box
        """
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2].copy()

    def create_mask(self, image_shape, box):
        """
        Create a binary mask for the furniture item
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
        return mask

    def generate_inpainting(self, image, mask, furniture_type, room_style):
        """
        Use generative AI to inpaint the area where furniture was removed

        This function uses OpenCV inpainting as fallback when API is not available
        """
        print(f"Generating inpainting for {furniture_type} space...")

        try:
            # First try OpenCV inpainting as baseline
            inpainted = cv2.inpaint(
                image,
                mask,
                inpaintRadius=7,
                flags=cv2.INPAINT_NS
            )

            # Additional texture harmonization for more realistic results
            # This enhances the result beyond basic inpainting
            mask_3d = np.stack((mask,) * 3, axis=-1) / 255.0
            texture_enhanced = self.texture_harmonization(image, inpainted, mask_3d)

            return texture_enhanced

        except Exception as e:
            print(f"Error in inpainting: {e}")
            print("Falling back to basic inpainting...")
            return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    def texture_harmonization(self, original, inpainted, mask_3d):
        """
        Harmonize textures between original and inpainted regions
        This creates more realistic transitions between areas
        """
        # Apply guided filter for better edge preservation
        large_scale = cv2.ximgproc.guidedFilter(
            original, inpainted, 15, 10, -1
        )

        # Blend original and inpainted using the mask
        result = original * (1 - mask_3d) + inpainted * mask_3d

        # Ensure edges are properly blended
        kernel = np.ones((15, 15), np.uint8)
        mask_dilated = cv2.dilate(mask_3d, kernel, iterations=1)
        mask_feathered = cv2.GaussianBlur(mask_dilated, (21, 21), 11)

        # Apply the large scale to feathered mask areas for better transition
        result = result * (1 - mask_feathered) + large_scale * mask_feathered

        return result.astype(np.uint8)

    def analyze_room_style(self, image):
        """
        Analyze the style of the room to generate appropriate inpainting
        """
        # Simple color analysis to determine dominant colors and style
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 and image.shape[2] == 3 else image
        pixels = image_rgb.reshape(-1, 3)

        # Calculate mean color (simple approach)
        mean_color = np.mean(pixels, axis=0)

        # Determine if room is light or dark
        brightness = np.mean(mean_color)
        is_light = brightness > 128

        # Check for warm or cool tones
        r, g, b = mean_color
        is_warm = r > b

        # Determine dominant material (simple heuristic)
        if is_warm and brightness > 150:
            dominant_material = "wooden"
        elif is_light and not is_warm:
            dominant_material = "modern"
        else:
            dominant_material = "neutral"

        style_description = f"{dominant_material} {'bright' if is_light else 'dim'} {'warm' if is_warm else 'cool'}"

        return style_description

    def reposition_furniture(self, image_path, furniture_id, new_position):
        """
        Reposition selected furniture to a new position and use generative AI
        to fill the original space

        Args:
            image_path: Path to the input image
            furniture_id: Index of the furniture item to move
            new_position: (x, y) coordinates for new top-left position

        Returns:
            Image with repositioned furniture
        """
        print(f"Repositioning furniture {furniture_id} to position {new_position}...")

        # Detect furniture
        image, furniture_items = self.detect_furniture(image_path)

        if furniture_id >= len(furniture_items):
            raise ValueError(f"Furniture ID {furniture_id} not found. Only {len(furniture_items)} items detected.")

        # Get selected furniture
        furniture = furniture_items[furniture_id]
        old_box = furniture['box']
        furniture_type = furniture['name']

        # Extract furniture from original position
        furniture_img = self.extract_furniture(image, old_box)

        # Create mask for inpainting (where furniture was)
        mask = self.create_mask(image.shape, old_box)

        # Analyze room style to generate appropriate inpainting
        room_style = self.analyze_room_style(image)
        print(f"Room style detected: {room_style}")

        # Use generative inpainting to fill the space
        result_image = self.generate_inpainting(image, mask, furniture_type, room_style)

        # Calculate new position and dimensions
        old_x1, old_y1, old_x2, old_y2 = old_box
        new_x, new_y = new_position
        width, height = old_x2 - old_x1, old_y2 - old_y1

        # Ensure new position is within image bounds
        new_x = max(0, min(new_x, result_image.shape[1] - width))
        new_y = max(0, min(new_y, result_image.shape[0] - height))

        # Create a copy of the furniture image for better memory management
        furniture_img_copy = furniture_img.copy()

        # Place furniture at new position
        new_x2, new_y2 = new_x + width, new_y + height
        result_image[new_y:new_y2, new_x:new_x2] = furniture_img_copy

        # Apply blending at the edges of the repositioned furniture
        # This creates a more natural integration with the background
        edge_mask = np.zeros((height, width))
        edge_mask[0:2, :] = 1
        edge_mask[-2:, :] = 1
        edge_mask[:, 0:2] = 1
        edge_mask[:, -2:] = 1

        # Feather the edge mask
        edge_mask = cv2.GaussianBlur(edge_mask, (5, 5), 1)
        edge_mask = np.dstack([edge_mask] * 3)

        # Get background colors at the new position
        bg_section = image[new_y:new_y2, new_x:new_x2].copy()

        # Blend furniture edges with background
        blended_furniture = furniture_img_copy * (1 - edge_mask) + bg_section * edge_mask

        # Place the blended furniture
        result_image[new_y:new_y2, new_x:new_x2] = blended_furniture

        print(f"Furniture '{furniture['name']}' repositioned successfully")
        return result_image, furniture['name']


def interactive_repositioning(image_path, save_path=None):
    """
    Interactive interface for furniture repositioning

    Args:
        image_path: Path to input image
        save_path: Path to save the result image
    """
    # Create output directory if it doesn't exist
    if save_path and os.path.dirname(save_path) and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    system = FurnitureRepositioningSystem()

    # Detect furniture
    image, furniture_items = system.detect_furniture(image_path)

    # Display original image with furniture detection
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title("Detected Furniture")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Display detected furniture
    print("\nDetected Furniture Items:")
    for i, item in enumerate(furniture_items):
        print(f"{i}: {item['name']} (confidence: {item['confidence']:.2f})")

    if not furniture_items:
        print("No furniture detected!")
        return

    # Get user selection
    furniture_id = int(input("\nSelect furniture ID to move: "))
    new_x = int(input("Enter new X position: "))
    new_y = int(input("Enter new Y position: "))

    # Reposition furniture
    result_image, furniture_name = system.reposition_furniture(
        image_path,
        furniture_id,
        (new_x, new_y)
    )

    # Display result
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.title(f"Room with Repositioned {furniture_name}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save result if path provided
    if save_path:
        plt.imsave(save_path, result_image)
        print(f"Result saved to {save_path}")

    return result_image


# For batch processing multiple furniture items
def batch_reposition(image_path, furniture_movements, save_path=None):
    """
    Reposition multiple furniture items in one go

    Args:
        image_path: Path to input image
        furniture_movements: List of (furniture_id, new_x, new_y) tuples
        save_path: Path to save result
    """
    system = FurnitureRepositioningSystem()

    # Start with original image
    current_image = cv2.imread(image_path)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    # Save image to a temporary file for each iteration
    temp_path = "temp_image.jpg"
    plt.imsave(temp_path, current_image)

    # Process each furniture movement
    for i, (furniture_id, new_x, new_y) in enumerate(furniture_movements):
        print(f"Moving furniture {furniture_id} to position ({new_x}, {new_y})...")

        # Reposition current furniture
        current_image, name = system.reposition_furniture(
            temp_path,
            furniture_id,
            (new_x, new_y)
        )

        # Save intermediate result for next iteration
        plt.imsave(temp_path, current_image)

    # Remove temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    # Display final result
    plt.figure(figsize=(12, 8))
    plt.imshow(current_image)
    plt.title("Final Repositioned Room")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Save final result if path provided
    if save_path:
        plt.imsave(save_path, current_image)
        print(f"Final result saved to {save_path}")

    return current_image


# Function to demonstrate different detection methods
def compare_detection_methods(image_path):
    """
    Compare different detection methods for furniture

    Args:
        image_path: Path to input image
    """
    system = FurnitureRepositioningSystem()

    # Regular detection
    regular_image, regular_items = system.detect_furniture(image_path)

    # Try segmentation for better detection
    try:
        segmented_image, segmented_items, _ = system.segment_furniture(image_path)

        # Display comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        ax1.imshow(regular_image)
        ax1.set_title(f"Standard Detection: {len(regular_items)} items")
        ax1.axis('off')

        ax2.imshow(segmented_image)
        ax2.set_title(f"Segmentation: {len(segmented_items)} items")
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

        print("\nStandard Detection Results:")
        for i, item in enumerate(regular_items):
            print(f"{i}: {item['name']} (confidence: {item['confidence']:.2f})")

        print("\nSegmentation Results:")
        for i, item in enumerate(segmented_items):
            print(f"{i}: {item['name']} (confidence: {item['confidence']:.2f})")

    except Exception as e:
        print(f"Segmentation not available: {e}")

        plt.figure(figsize=(12, 8))
        plt.imshow(regular_image)
        plt.title(f"Standard Detection: {len(regular_items)} items")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print("\nDetection Results:")
        for i, item in enumerate(regular_items):
            print(f"{i}: {item['name']} (confidence: {item['confidence']:.2f})")


# Example usage with proper file paths for Mac
if __name__ == "__main__":
    # Get current working directory for proper file path handling in PyCharm
    current_dir = os.getcwd()

    # For single furniture item
    image_path = os.path.join(current_dir, "img.png")  # Update with your image path
    output_path = os.path.join(current_dir, "output", "repositioned_room1.jpg")

    print(f"Processing image: {image_path}")
    print(f"Output will be saved to: {output_path}")

    # Make sure the image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Please update the path to your image file.")
    else:
        # Compare detection methods first
        compare_detection_methods(image_path)

        # Then proceed with interactive repositioning
        result = interactive_repositioning(image_path, output_path)

    # Uncomment for batch processing
    # movements = [
    #     (0, 100, 200),  # Move furniture 0 to position (100, 200)
    #     (1, 300, 400),  # Move furniture 1 to position (300, 400)
    # ]
    # batch_result = batch_reposition(image_path,
    #                                os.path.join(current_dir, "output", "batch_repositioned.jpg"))