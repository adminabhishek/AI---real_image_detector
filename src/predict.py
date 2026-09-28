"""
Run inference on a single image using the trained model.

Usage:
    python src/predict.py path/to/image.jpg
"""
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import build_model
from src.dataset import eval_transform
from PIL import Image

CHECKPOINT_PATH = "checkpoints/best_model.pth"


def predict_image(image_path, model=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = build_model(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        logit = model(tensor)
        prob_fake = torch.sigmoid(logit).item()

    verdict = "FAKE" if prob_fake > 0.5 else "REAL"
    return verdict, prob_fake


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py path/to/image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    verdict, prob_fake = predict_image(image_path)
    print(f"\nImage: {image_path}")
    print(f"Verdict: {verdict}")
    print(f"P(fake) = {prob_fake:.4f}  |  Confidence = {max(prob_fake, 1-prob_fake)*100:.1f}%")
