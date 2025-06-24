import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from model_architecture import AstronomyCNN  # Replace this with your actual model class name

# Two classes: Star and Galaxy
class_names = ['Galaxy', 'Star']  # Make sure this matches your training label order

def load_model(model_path):
    model = AstronomyCNN()  # Replace with your actual class name and args if needed
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

'''
def predict_class(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Match your model's input size
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # For RGB images
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return class_names[predicted.item()], confidence.item()
'''
def predict_class(image: Image.Image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize input to 224x224 as expected by the model
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    class_names = ['Galaxy', 'Star']  # Ensure matches your classes
    return class_names[predicted.item()], confidence.item()


