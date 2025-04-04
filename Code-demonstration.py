import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
model = models.resnet50(pretrained=True)
model.eval()  # Set to evaluation mode
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

image_path = "your_image.jpg"  # Replace with your image
input_image = preprocess_image(image_path)
def generate_random_masks(size, num_masks=1000, mask_prob=0.5):
    masks = np.random.rand(num_masks, size, size) < mask_prob
    return torch.tensor(masks, dtype=torch.float32)

mask_size = 7  # Size of the occlusion
num_masks = 5000  # More masks give better explanations
masks = generate_random_masks(224, num_masks)
def apply_masks(image, masks):
    masked_images = image * masks.view(-1, 1, 224, 224)
    return masked_images

masked_images = apply_masks(input_image, masks)
@torch.no_grad()
def compute_rise_saliency(model, image, masks):
    outputs = model(masked_images)  # Get model predictions
    scores = torch.nn.functional.softmax(outputs, dim=1)[:, torch.argmax(outputs)]  # Extract scores for top class
    importance_map = torch.matmul(scores, masks.view(num_masks, -1)).reshape(224, 224) / num_masks
    return importance_map.numpy()

saliency_map = compute_rise_saliency(model, input_image, masks)

# Normalize and Display Saliency Map
saliency_map = cv2.applyColorMap((saliency_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
plt.imshow(saliency_map)
plt.axis("off")
plt.show()