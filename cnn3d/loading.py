import torch
from cnn3d.models.cnn3d_own import Cnn3DOwn
from cnn3d.models.cnn3d_elassy import Cnn3DElAssy
from cnn3d.models.cnn3d_googlenet import Cnn3DGoogleNet


# Function to load the trained model
def load_model(model_path, device):
    
    selected_model = int(model_path.split('/')[-1].split('-')[1].split('_')[0])

    if selected_model == 1:
        model = Cnn3DOwn().to(device)
    elif selected_model == 2:
        model = Cnn3DElAssy().to(device)
    elif selected_model == 3:
        model = Cnn3DGoogleNet().to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model