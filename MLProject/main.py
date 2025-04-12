import streamlit as st
import torch
import cv2
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
from ultralytics import YOLO

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8x for enhanced object detection
object_detector = YOLO("yolov8x.pt").to(device)

# Load Stable Diffusion Inpainting model
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

st.title("Object Removal from Furnished Room")

uploaded_file = st.file_uploader("Upload Furnished Room Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_cv = np.array(image, dtype=np.float32)  # Convert to float32 for YOLO
    image_cv_bgr = cv2.cvtColor(image_cv.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Detect objects with enhanced settings
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
    draw = ImageDraw.Draw(image)
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), obj["label"], fill="red")

    st.image(image, caption="Detected Objects - Select to Remove", use_container_width=True)

    if detected_objects:
        object_labels = [f"{obj['label']} ({i+1})" for i, obj in enumerate(detected_objects)]
        selected_index = st.selectbox("Select Object to Remove", range(len(object_labels)), format_func=lambda i: object_labels[i])

        if st.button("Remove Selected Object"):
            object_bbox = detected_objects[selected_index]["bbox"]
            mask = np.full((image.height, image.width), 0, dtype=np.uint8)
            x1, y1, x2, y2 = object_bbox
            mask[y1:y2, x1:x2] = 255  # Fill mask with white where the object is
            mask_image = Image.fromarray(mask)

            with torch.no_grad():
                result = inpaint_pipe(prompt="Realistic room without the object", image=image, mask_image=mask_image).images[0]
            st.image(result, caption="Edited Room Image", use_column_width=True)
    else:
        st.warning("No objects detected. Try a different image or settings.")
