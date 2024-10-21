import torch

import numpy as np

from tools.preprocess_image import preprocess_image

# Function to make a prediction
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1) # Convert logits to probabilities
        
    return probabilities

# Function to predict the class of a .nii image using the trained model
def predict(model, device, image_path):
    
    # Preprocess the image
    image_np = preprocess_image(image_path, labels=False)

    # Check if the preprocessing function adds channel dimension
    if image_np.ndim == 3:
        image_np = np.expand_dims(image_np, axis=0)  # Add channel dimension: [1, depth, height, width]
    
    # Convert the NumPy array to a PyTorch tensor and add batch dimension
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)
    
    # Make a prediction
    probabilities_tensor = predict_image(model, image_tensor)
    
    # Convert tensor to list
    probabilities = probabilities_tensor.cpu().numpy().flatten().tolist()
    
    return probabilities
