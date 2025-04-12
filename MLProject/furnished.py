import torch
import cv2
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import streamlit as st
import os
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Room Furnishing App", layout="wide")


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


# Main application layout
st.title("Virtual Interior Designing 🏠")
st.markdown("Upload an unfurnished room image and transform it into a beautifully furnished space.")

# Sidebar for model settings
with st.sidebar:
    st.header("Settings")
    num_steps = st.slider("Inference Steps", min_value=20, max_value=50, value=30)

    style_options = [
        "A beautifully furnished room with modern furniture, warm lighting, and elegant decor.",
        "Modern style with sleek furniture and warm lighting",
        "Classic vintage style with wooden furniture and antique decor",
        "Minimalist style with clean lines and neutral colors",
        "Luxury hotel suite style with plush seating and ambient lighting",
        "Bohemian style with colorful patterns and artistic decor"
    ]

    prompt = st.selectbox("Choose a Furnishing Style", style_options)

    if st.button("Load Model", type="primary"):
        with st.spinner("Loading model..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_mlsd", torch_dtype=dtype).to(
                device)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                controlnet=controlnet,
                torch_dtype=dtype
            ).to(device)

            if torch.cuda.is_available():
                pipe.enable_xformers_memory_efficient_attention()

            st.session_state['pipe'] = pipe
            st.success("✅ Model loaded successfully!")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Unfurnished Room")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_path = save_uploaded_image(uploaded_file)
        st.image(img_path, caption="Original Room", use_container_width=True)

with col2:
    st.header("Furnished Result")

    if st.button("🪄 Generate Furnished Room", type="primary", use_container_width=True):
        if 'pipe' in st.session_state and uploaded_file is not None:
            with st.spinner("🔄 Generating furnished room design..."):
                img_path = os.path.join('temp', uploaded_file.name)
                init_image = Image.open(img_path).convert("RGB")

                try:
                    with torch.no_grad():
                        pipe = st.session_state['pipe']
                        result = pipe(
                            prompt=prompt,
                            image=init_image,
                            num_inference_steps=30
                        ).images[0]

                    result_path = os.path.join('temp', f"furnished_{uploaded_file.name}")
                    result.save(result_path)
                    st.image(result_path, caption="Furnished Room", use_container_width=True)

                    with open(result_path, "rb") as file:
                        st.download_button(
                            label="Download Furnished Room",
                            data=file,
                            file_name=f"furnished_{uploaded_file.name}",
                            mime=f"image/{uploaded_file.name.split('.')[-1]}"
                        )
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")
        else:
            st.warning("Please upload an image and load the model first.")

st.markdown("---")
st.caption("Room Furnishing App powered by Stable Diffusion and ControlNet")
