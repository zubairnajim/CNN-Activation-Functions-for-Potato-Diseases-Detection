import torch
from torchvision import transforms
from PIL import Image
from latest import initialize_model  # Import the initialize_model function from your main script

# Configuration
model_name = "resnet18"  # Change to the model you want to evaluate
activation_name = "prelu"  # Change to the activation function used during training
model_path = "best_resnet18_prelu.pt"  # Path to the saved model
image_path = r"C:\Drive D\MSc\Deep Learning\Potato Classification\potato_disease_dataset\Dry Rot\13.jpg"  # Path to the image you want to classify
class_names = ["Black Scurf", "Blackleg", "Commong Scab", "Dry Rot", "Healthy Potatoes", "Miscellaneous", "Pink Pot"]  # Replace with your actual class names

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the trained model
num_classes = len(class_names)
model = initialize_model(model_name, activation_name, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set the model to evaluation mode
model.to(device)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert('RGB')
input_tensor = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Run inference
with torch.no_grad():
    output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)

# Get the class name
predicted_label = class_names[predicted_class.item()]
print(f"The image is classified as: {predicted_label}")