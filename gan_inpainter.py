import cv2
import numpy as np
import torch
import os

# Try importing diffusers pipeline for inpainting
try:
    from diffusers import StableDiffusionInpaintPipeline
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

class GANInpainter:
    def __init__(self, device=None):
        """Initialize the inpainting model (Stable Diffusion Inpainting) if available."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.use_gan = False
        if DIFFUSERS_AVAILABLE:
            try:
                # Load Stable Diffusion Inpainting model
                print("Loading Stable Diffusion inpainting model...")
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting", revision="fp16", torch_dtype=torch.float16
                )
                self.pipe = self.pipe.to(self.device)
                # Optionally disable safety checker
                if hasattr(self.pipe, 'safety_checker'):
                    self.pipe.safety_checker = lambda images, **kwargs: (images, False)
                print("Stable Diffusion inpainting model loaded.")
                self.use_gan = True
            except Exception as e:
                print(f"Error loading GAN inpainting model: {e}\nFalling back to OpenCV inpainting.")
                self.use_gan = False
        else:
            print("Diffusers not installed; using OpenCV inpainting.")

    def inpaint(self, image, mask):
        """Run inpainting on the image with the given binary mask."""
        # mask: single-channel numpy 0/255
        if self.use_gan:
            try:
                # Convert to PIL images
                img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                mask_pil = Image.fromarray(mask)
                # Run pipeline
                result = self.pipe(image=img_pil, mask_image=mask_pil).images[0]
                # Convert back to BGR numpy
                out = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                return out
            except Exception as e:
                print(f"GAN inpainting failed: {e}\nFalling back to OpenCV.")
        # Fallback: OpenCV Telea inpainting
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        return inpainted 