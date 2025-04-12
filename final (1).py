import streamlit as st
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import tempfile
import io
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel,StableDiffusionInpaintPipeline
from ultralytics import YOLO

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


# Page configuration
st.set_page_config(
    page_title="Interior Room Designer",
    page_icon="🪑",
    layout="wide"
)

# Add custom CSS styling
st.markdown("""
<style>
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #3B82F6;
        margin-bottom: 0.5rem;
    }
    .info-text {
        background-color: #8c3c2b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-text {
        background-color: #DCFCE7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .feature-box {
        border: 1px solid #E5E7EB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Interior Room Designer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="info-text">Complete room design solution: generate a completely new furnished room design or Upload a room image to detect and reposition existing furniture.</div>',
    unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'system' not in st.session_state:
    st.session_state.system = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'furniture_items' not in st.session_state:
    st.session_state.furniture_items = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'temp_image_path' not in st.session_state:
    st.session_state.temp_image_path = None
if 'sd_model_loaded' not in st.session_state:
    st.session_state.sd_model_loaded = False


# Function to convert OpenCV image to a format Streamlit can display
def convert_image(image):
    if image is None:
        return None

    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# Function to resize image
def resize_image(image, size=(512, 512)):
    return image.resize(size)


# Function to save uploaded image
def save_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        os.makedirs('temp', exist_ok=True)
        img = Image.open(uploaded_file)
        img = resize_image(img)
        img_path = os.path.join('temp', uploaded_file.name)
        img.save(img_path)
        return img_path
    return None


# Create tabs for the different features
tab1, tab2, tab3, tab4,tab5 = st.tabs(["Generate New Design","Upload & Detect", "Reposition", "Compare","Remove"])

# Tab 1: Generate New Design
with tab1:
    st.markdown('<div class="subheader">Generate a Completely New Room Design</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.write("### Upload Unfurnished Room")
        uploaded_file_sd = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="sd_upload")
        
        if uploaded_file_sd is not None:
            img_path = save_uploaded_image(uploaded_file_sd)
            st.image(img_path, caption="Original Room", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model settings
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.write("### Design Settings")
        
        num_steps = st.slider("Inference Steps", min_value=20, max_value=50, value=30)
        
        style_options = [
            "Minimalist Design Style",
            "Modern Design Style",
            "Vintage Design Style",
            "Traditional Design Style",
            "Scandinavian Design Style"
        ]
        
        room_options = ["Living Room", "Dining Room", "Bedroom", "Bathroom", "Kitchen"]
        
        style = st.selectbox("Choose a Furnishing Style", style_options)
        room_type = st.selectbox("Choose the type of Room", room_options)
        
        # Load model button
        if st.button("Load AI Model", type="primary"):
            with st.spinner("Loading model..."):
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    
                    controlnet = ControlNetModel.from_pretrained(
                        "lllyasviel/control_v11p_sd15_mlsd", 
                        torch_dtype=dtype
                    ).to(device)
                    
                    pipe = StableDiffusionControlNetPipeline.from_pretrained(
                        "runwayml/stable-diffusion-v1-5",
                        controlnet=controlnet,
                        torch_dtype=dtype
                    ).to(device)
                    
                    if torch.cuda.is_available():
                        pipe.enable_xformers_memory_efficient_attention()
                    
                    st.session_state['pipe'] = pipe
                    st.session_state.sd_model_loaded = True
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.write("### Furnished Result")
        
        # Generate button
        generate_button = st.button("🪄 Generate Furnished Room Design", type="primary", use_container_width=True)
        
        if generate_button:
            if st.session_state.sd_model_loaded and uploaded_file_sd is not None:
                with st.spinner("🔄 Generating furnished room design..."):
                    try:
                        img_path = os.path.join('temp', uploaded_file_sd.name)
                        init_image = Image.open(img_path).convert("RGB")
                        
                        with torch.no_grad():
                            pipe = st.session_state['pipe']
                            result = pipe(
                                prompt=f"A beautifully furnished {room_type} with {style} Furnitures, warm lighting, and elegant decor.",
                                image=init_image,
                                num_inference_steps=num_steps
                            ).images[0]
                        
                        result_path = os.path.join('temp', f"furnished_{uploaded_file_sd.name}")
                        result.save(result_path)
                        st.image(result_path, caption="AI Generated Furnished Room", use_container_width=True)
                        
                        with open(result_path, "rb") as file:
                            st.download_button(
                                label="Download Generated Design",
                                data=file,
                                file_name=f"furnished_{uploaded_file_sd.name}",
                                mime=f"image/{uploaded_file_sd.name.split('.')[-1]}"
                            )
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
            else:
                st.warning("Please upload an image and load the AI model first.")
        else:
            st.info("Upload an image and load the AI model, then click Generate to create a new design")
        st.markdown('</div>', unsafe_allow_html=True)


# Tab 2: Upload and Detect Furniture
with tab2:
    st.markdown('<div class="subheader">Step 1: Upload a Room Image</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="detect_upload")

    if uploaded_file is not None:
        # Initialize the system if not already done
        if st.session_state.system is None:
            with st.spinner("Initializing furniture detection system..."):
                st.session_state.system = FurnitureRepositioningSystem()
                st.success("System initialized!")

        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            st.session_state.temp_image_path = temp_file.name

        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Furniture detection button
        if st.button("Detect Furniture"):
            with st.spinner("Detecting furniture..."):
                try:
                    # Use the existing detect_furniture method
                    processed_image, furniture_items = st.session_state.system.detect_furniture(
                        st.session_state.temp_image_path)

                    # Save results to session state
                    st.session_state.processed_image = processed_image
                    st.session_state.furniture_items = furniture_items

                    # Read original image for later use
                    original_image = cv2.imread(st.session_state.temp_image_path)
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                    st.session_state.original_image = original_image

                    # Display success message
                    st.markdown(f'<div class="success-text">Detected {len(furniture_items)} furniture items!</div>',
                                unsafe_allow_html=True)

                    # Display detected furniture image
                    st.image(convert_image(processed_image), caption="Detected Furniture", use_column_width=True)

                    # Display furniture list
                    st.markdown('<div class="subheader">Detected Furniture Items:</div>', unsafe_allow_html=True)
                    for i, item in enumerate(furniture_items):
                        st.write(f"{i}: {item['name']} (confidence: {item['confidence']:.2f})")

                except Exception as e:
                    st.error(f"Error detecting furniture: {str(e)}")

        # Option to try segmentation for better detection
        if st.checkbox("Try Segmentation for Better Detection"):
            with st.spinner("Performing furniture segmentation..."):
                try:
                    # Use the existing segment_furniture method
                    segmented_image, segmented_items, masks = st.session_state.system.segment_furniture(
                        st.session_state.temp_image_path)

                    # Display segmented image
                    st.image(convert_image(segmented_image), caption="Segmented Furniture", use_column_width=True)

                    # Display segmented furniture list
                    st.markdown('<div class="subheader">Segmented Furniture Items:</div>', unsafe_allow_html=True)
                    for i, item in enumerate(segmented_items):
                        st.write(f"{i}: {item['name']} (confidence: {item['confidence']:.2f})")

                    # Option to use segmentation results instead
                    if st.button("Use Segmentation Results"):
                        st.session_state.processed_image = segmented_image
                        st.session_state.furniture_items = segmented_items
                        st.success("Now using segmentation results!")

                except Exception as e:
                    st.error(f"Error performing segmentation: {str(e)}")

# Tab 3: Reposition Furniture
with tab3:
    st.markdown('<div class="subheader">Step 2: Reposition Furniture</div>', unsafe_allow_html=True)

    if st.session_state.furniture_items is None:
        st.info("Please upload an image and detect furniture first (in the Upload & Detect tab)")
    else:
        # Display the image with detected furniture
        st.image(convert_image(st.session_state.processed_image), caption="Detected Furniture", use_column_width=True)

        # Create columns for input fields
        col1, col2 = st.columns(2)

        with col1:
            # Furniture selection dropdown
            furniture_options = [f"{i}: {item['name']}" for i, item in enumerate(st.session_state.furniture_items)]
            selected_furniture = st.selectbox("Select furniture to move:", furniture_options)
            furniture_id = int(selected_furniture.split(":")[0])

            # Display current position
            current_box = st.session_state.furniture_items[furniture_id]['box']
            st.write(
                f"Current position: (x1={current_box[0]}, y1={current_box[1]}, x2={current_box[2]}, y2={current_box[3]})")

        with col2:
            # Get image dimensions
            img_height, img_width = st.session_state.original_image.shape[:2]

            # New position inputs
            new_x = st.number_input("New X position:", 0, img_width, current_box[0], step=10)
            new_y = st.number_input("New Y position:", 0, img_height, current_box[1], step=10)

        # Reposition button
        if st.button("Reposition Furniture"):
            with st.spinner("Repositioning furniture..."):
                try:
                    # Save current processed image to temp file for repositioning
                    temp_processed_path = os.path.join(tempfile.gettempdir(), "processed_image.jpg")
                    plt.imsave(temp_processed_path, st.session_state.processed_image)

                    # Use the existing reposition_furniture method
                    result_image, furniture_name = st.session_state.system.reposition_furniture(
                        temp_processed_path,
                        furniture_id,
                        (new_x, new_y)
                    )

                    # Save result to session state
                    st.session_state.result_image = result_image

                    # Display result
                    st.markdown(f'<div class="success-text">Successfully repositioned {furniture_name}!</div>',
                                unsafe_allow_html=True)
                    st.image(convert_image(result_image), caption=f"Room with Repositioned {furniture_name}",
                             use_column_width=True)

                    # Clean up temp file
                    if os.path.exists(temp_processed_path):
                        os.remove(temp_processed_path)

                except Exception as e:
                    st.error(f"Error repositioning furniture: {str(e)}")

# Tab 4: Compare Before/After
with tab4:
    st.markdown('<div class="subheader">Step 3: Compare Results</div>', unsafe_allow_html=True)

    if st.session_state.result_image is None:
        st.info("Please reposition furniture first (in the Reposition tab)")
    else:
        # Display before and after images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Room**", unsafe_allow_html=True)
            st.image(convert_image(st.session_state.original_image), use_column_width=True)

        with col2:
            st.markdown("**Redesigned Room**", unsafe_allow_html=True)
            st.image(convert_image(st.session_state.result_image), use_column_width=True)

        # Option to download result
        result_img = Image.fromarray(convert_image(st.session_state.result_image).astype('uint8'), 'RGB')
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        st.download_button(
            label="Download Result",
            data=buf.getvalue(),
            file_name="repositioned_room.png",
            mime="image/png"
        )

        # Option to try with different furniture
        if st.button("Try with Different Furniture"):
            st.session_state.processed_image = st.session_state.result_image
            st.experimental_rerun()

        # Option to try batch repositioning
        st.markdown('<div class="subheader">Batch Repositioning</div>', unsafe_allow_html=True)
        st.write("You can also reposition multiple furniture items at once:")

        # Create a text area for batch movement input
        batch_input = st.text_area(
            "Enter furniture movements as 'furniture_id,new_x,new_y' (one per line):",
            height=100,
            help="Example:\n0,100,200\n1,300,400"
        )

        if st.button("Apply Batch Movements"):
            with st.spinner("Applying batch furniture movements..."):
                try:
                    # Parse batch input
                    movements = []
                    for line in batch_input.strip().split('\n'):
                        if line:
                            parts = line.split(',')
                            if len(parts) == 3:
                                furniture_id = int(parts[0])
                                new_x = int(parts[1])
                                new_y = int(parts[2])
                                movements.append((furniture_id, new_x, new_y))

                    if movements:
                        # Save current processed image to temp file for repositioning
                        temp_path = os.path.join(tempfile.gettempdir(), "batch_processed.jpg")
                        plt.imsave(temp_path, st.session_state.original_image)

                        # Use the existing batch_reposition method with our temp path
                        batch_result = st.session_state.system.batch_reposition(
                            temp_path,
                            movements,
                            os.path.join(tempfile.gettempdir(), "batch_result.jpg")
                        )

                        # Display batch result
                        st.session_state.result_image = batch_result
                        st.markdown('<div class="success-text">Batch repositioning completed!</div>',
                                    unsafe_allow_html=True)
                        st.image(convert_image(batch_result), caption="Room with Multiple Repositioned Items",
                                 use_column_width=True)

                        # Clean up temp files
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    else:
                        st.warning("No valid movements specified. Please check your input format.")

                except Exception as e:
                    st.error(f"Error in batch repositioning: {str(e)}")

# Tab 5: Remove Objects
with tab5:
    st.markdown('<div class="subheader">Remove Objects from Room</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-text">Upload a room image, detect objects, and remove unwanted items with AI inpainting technology.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.write("### Upload Room Image")
        uploaded_file_remove = st.file_uploader("Upload Furnished Room Image", type=["png", "jpg", "jpeg"], key="remove_upload")
        
        # Load models button
        if st.button("Load Object Detection & Inpainting Models", type="primary"):
            with st.spinner("Loading models..."):
                try:
                    # Set device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # Load YOLOv8x for enhanced object detection
                    object_detector = YOLO("yolov8x.pt").to(device)
                    st.session_state['object_detector'] = object_detector
                    st.session_state.object_detector_loaded = True
                    
                    # Load Stable Diffusion Inpainting model
                    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        "runwayml/stable-diffusion-inpainting",
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    ).to(device)
                    st.session_state['inpaint_pipe'] = inpaint_pipe
                    st.session_state.inpaint_pipe_loaded = True
                    
                    st.success("✅ Object detection and inpainting models loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display uploaded image and detected objects
        if uploaded_file_remove:
            image = Image.open(uploaded_file_remove).convert("RGB")
            # Save image to temp directory for processing
            os.makedirs('temp', exist_ok=True)
            img_path = os.path.join('temp', uploaded_file_remove.name)
            image.save(img_path)
            
            # Convert to arrays for detection
            image_cv = np.array(image, dtype=np.float32)
            image_cv_bgr = cv2.cvtColor(image_cv.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            if st.session_state.object_detector_loaded:
                with st.spinner("Detecting objects..."):
                    try:
                        # Detect objects with enhanced settings
                        object_detector = st.session_state['object_detector']
                        results = object_detector(image_cv_bgr, conf=0.1, imgsz=1280, iou=0.4, augment=True)
                        
                        detected_objects = []
                        for r in results:
                            for box in r.boxes.data:
                                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                                detected_objects.append({
                                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                                    "label": object_detector.names[int(cls)]
                                })
                        
                        # Draw bounding boxes
                        image_with_boxes = image.copy()
                        draw = ImageDraw.Draw(image_with_boxes)
                        for obj in detected_objects:
                            x1, y1, x2, y2 = obj["bbox"]
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            draw.text((x1, y1), obj["label"], fill="red")
                        
                        st.image(image_with_boxes, caption="Detected Objects - Select to Remove", use_container_width=True)
                        
                        # Store in session state
                        st.session_state['detected_objects'] = detected_objects
                        st.session_state['removal_image'] = image
                    except Exception as e:
                        st.error(f"Error detecting objects: {str(e)}")
            else:
                st.warning("Please load the object detection model first.")
    
    with col2:
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        st.write("### Select & Remove Objects")
        
        if 'detected_objects' in st.session_state and st.session_state['detected_objects']:
            detected_objects = st.session_state['detected_objects']
            
            # Create selection dropdown
            object_labels = [f"{obj['label']} ({i+1})" for i, obj in enumerate(detected_objects)]
            selected_index = st.selectbox("Select Object to Remove", range(len(object_labels)), format_func=lambda i: object_labels[i])
            
            # Remove button
            if st.button("Remove Selected Object"):
                if st.session_state.inpaint_pipe_loaded:
                    with st.spinner("Removing object and inpainting..."):
                        try:
                            image = st.session_state['removal_image']
                            object_bbox = detected_objects[selected_index]["bbox"]
                            
                            # Create mask
                            mask = np.zeros((image.height, image.width), dtype=np.uint8)
                            x1, y1, x2, y2 = object_bbox
                            mask[y1:y2, x1:x2] = 255  # Fill mask with white where the object is
                            mask_image = Image.fromarray(mask)
                            
                            # Inpaint
                            with torch.no_grad():
                                inpaint_pipe = st.session_state['inpaint_pipe']
                                result = inpaint_pipe(
                                    prompt="Realistic room without the object", 
                                    image=image, 
                                    mask_image=mask_image
                                ).images[0]
                            
                            # Save and display result
                            result_path = os.path.join('temp', f"removed_object_{selected_index}.png")
                            result.save(result_path)
                            st.image(result, caption="Room with Object Removed", use_container_width=True)
                            
                            # Download button
                            with open(result_path, "rb") as file:
                                st.download_button(
                                    label="Download Edited Image",
                                    data=file,
                                    file_name=f"room_object_removed.png",
                                    mime="image/png"
                                )
                        except Exception as e:
                            st.error(f"Error removing object: {str(e)}")
                else:
                    st.warning("Please load the inpainting model first.")
        else:
            st.info("Please upload an image and detect objects first.")
        st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with app information
with st.sidebar:
    st.markdown('<div class="subheader">About This App</div>', unsafe_allow_html=True)
    st.write("""
    This application combines two powerful furniture design tools:
    
    1. **Room Designer**: Generate completely new room designs with different furnishing styles
    2. **Furniture Repositioning System**: Detect and move existing furniture in your room photos
    
    
    Choose the tab that best fits your needs!
    """)
    
    st.markdown('<div class="subheader">How to Use</div>', unsafe_allow_html=True)
    st.write("""
    **For New Designs:**
    1. Go to the fourth tab
    2. Upload a room image
    3. Choose a style and room type
    4. Load the AI model
    5. Generate a new furnished design

    **For Repositioning:**
    1. Upload a room image in the first tab
    2. Detect furniture items
    3. Select and reposition furniture in the second tab
    4. Compare results in the third tab
    
    """)


# Clean up temporary files when the app is closed
def cleanup():
    if st.session_state.temp_image_path and os.path.exists(st.session_state.temp_image_path):
        os.remove(st.session_state.temp_image_path)
    
    # Clean up SD temp files
    if os.path.exists('temp'):
        for file in os.listdir('temp'):
            os.remove(os.path.join('temp', file))


# Register cleanup function to be called when the app is closed
import atexit
atexit.register(cleanup)
