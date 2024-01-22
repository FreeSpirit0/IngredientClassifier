from ultralytics import YOLO

# Load a model
model = YOLO('model/v17.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['data/test/f.jpeg', 'data/test/f1.jpeg'], save=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs