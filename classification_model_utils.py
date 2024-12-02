import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

# Function to load the model with saved weights
def load_model(model_path, num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to preprocess an input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Same normalization used during training
    ])
    # Load the image and apply transformations
    image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB mode
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to predict the class of an image
def predict(image, model, class_names):
    image = preprocess_image(image)  # Preprocess the image
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)
    predicted_label = class_names[predicted_class.item()]
    return predicted_label
