from transformers import ViTImageProcessor, ViTForImageClassification
from datasets import IngredientsDataset, train_image_paths, train_transforms
import torch
from torch.utils.data import DataLoader
from PIL import Image
import requests

# url = 'https://vancouverwithlove.com/wp-content/uploads/2023/08/pasta-al-pesto-24-1.jpg'
# image = Image.open(requests.get(url, stream=True).raw)

image_processor = ViTImageProcessor.from_pretrained('nateraw/food', local_files_only=True)
model = ViTForImageClassification.from_pretrained('nateraw/food', local_files_only=True)
# inputs = image_processor(images=image, return_tensors="pt")

dataset = IngredientsDataset(train_image_paths, train_transforms)
train_loader = DataLoader(dataset=dataset, batch_size=32,shuffle=True)

print(train_loader.dataset[5])

# with torch.no_grad():
#     logits = model(**inputs).logits
# predicted_label = logits.argmax(-1).item()

# print(model.config.id2label[predicted_label])
