import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import torch.nn.functional as F


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

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            # try attention weights first
            if hasattr(outputs, "attentions") and outputs.attentions is not None:
                attn = outputs.attentions[-1].mean(dim=1)[0, 0, 1:]
            # fallback to hidden state magnitudes
            elif hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state[0, 1:]
                attn = torch.norm(hidden, dim=-1)
            else:
                raise ValueError("unsupported model output structure")

        # reshape to spatial grid
        attn = attn.cpu().numpy()
        grid_size = int(np.sqrt(len(attn)))
        attn = attn[: grid_size * grid_size].reshape(grid_size, grid_size)

        # normalize
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

        # upsample to image size
        return self._upsample(attn, image.size)

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
