import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, sobel, laplace


class SaliencyExtractor:
    """extract saliency maps using lightweight vision models"""

    def __init__(self, model_name="facebook/dinov2-small", device=None):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        print(f"loading {model_name} on {device}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def extract_saliency(self, image):
        """extract saliency from rgb image, returns normalized map [0,1]"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        # Convert to numpy for processing
        img_np = np.array(image)

        # Use traditional computer vision saliency methods
        # 1. Convert to grayscale
        gray = np.mean(img_np, axis=2)

        # 2. Gradient magnitude (edge detection) using scipy
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # 3. Laplacian for blob detection
        log = np.abs(laplace(gray))

        # 4. Combine gradient and Laplacian
        saliency = gradient_magnitude + 0.5 * log

        # 5. Apply Gaussian blur for smoothness
        saliency = gaussian_filter(saliency, sigma=1.0)

        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (
            saliency.max() - saliency.min() + 1e-8
        )

        # Resize to original image size using PIL
        saliency_pil = Image.fromarray((saliency * 255).astype(np.uint8))
        saliency_resized = saliency_pil.resize(image.size, Image.Resampling.LANCZOS)
        saliency_resized = np.array(saliency_resized) / 255.0

        return saliency_resized

    def extract_multi_view_saliency(self, views):
        """extract saliency from multiple views"""
        return [self.extract_saliency(v["image"]) for v in views]

    def _upsample(self, attn_map, target_size):
        """bilinear upsample to target resolution"""
        tensor = torch.from_numpy(attn_map).unsqueeze(0).unsqueeze(0).float()
        upsampled = F.interpolate(
            tensor, size=(target_size[1], target_size[0]), mode="bilinear"
        )
        return upsampled.squeeze().numpy()
