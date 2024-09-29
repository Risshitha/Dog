import torch
from torchvision import models, transforms
from PIL import Image

def classify_image(image_path, model):
    """
    Classifies an image using a pre-trained model and returns the predicted label index.
    """
    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get predicted label index
    _, predicted = output.max(1)
    return predicted.item()  # Return index of the class with the highest score
