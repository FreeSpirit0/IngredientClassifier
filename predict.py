from ultralytics import YOLO

# Load a model
model = YOLO('model/polygon/best (12).pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model([f'data/test/mix/m{i}.jpeg' for i in range(1, 23)], save=True)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs