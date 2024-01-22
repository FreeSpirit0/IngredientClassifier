from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8m.pt')

# Train the model using the 'data.yaml' dataset for 3 epochs
results = model.train(data='./ingredient/data.yaml', epochs=10, device='mps')

# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')