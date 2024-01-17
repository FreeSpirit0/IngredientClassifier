from transformers import ViTImageProcessor, ViTForImageClassification
import torch
from PIL import Image
import requests

url = 'https://vancouverwithlove.com/wp-content/uploads/2023/08/pasta-al-pesto-24-1.jpg'
image = Image.open(requests.get(url, stream=True).raw)

image_processor = ViTImageProcessor.from_pretrained('nateraw/food',)
model = ViTForImageClassification.from_pretrained('nateraw/food')
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits
predicted_label = logits.argmax(-1).item()

print(model.config.id2label[predicted_label])
