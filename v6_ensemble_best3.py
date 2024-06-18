import os
import pandas as pd
import torch
from torch import nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import multiprocessing

# Set higher precision for matrix multiplications
torch.set_float32_matmul_precision('high')

# Get the number of CPU cores
num_cpu_cores = multiprocessing.cpu_count()

# Define the path to your data
location = "detect-pneumonia-spring-2024/"  # Adjust the path accordingly

# Define the custom dataset
class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)  # Read the annotations from the CSV file
        self.root_dir = root_dir  # Directory with all the images
        self.transform = transform  # Transformation to be applied on images

    def __len__(self):
        return len(self.annotations)  # Total number of images in the dataset

    def __getitem__(self, idx):
        # Get image file path
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        # Open the image in grayscale mode
        image = Image.open(img_name).convert('L')
        # Get the label for the image
        label = int(self.annotations.iloc[idx, 1])

        # Apply the transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label

# Data augmentation and transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),  # Convert image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
])

# Load the models with their accuracies
model_paths = [
    (location + "models/v3_resnet18_gpu_s30k_b32_e100_p5_lr0001.pth", "resnet18", 3),
    (location + "models/v3_resnet50_gpu.pth", "resnet50", 1),
    (location + "models/v7_resnet50_gpu_s30k_b120_e50_p3.pth", "resnet50", 1)
]

# Define the custom model class
class PneumoniaModel(nn.Module):
    def __init__(self, model_name, num_classes=3, input_channels=1):
        super(PneumoniaModel, self).__init__()
        if model_name == "resnet18":
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)  # Load pretrained ResNet-18 model
        elif model_name == "resnet34":
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)  # Load pretrained ResNet-34 model
        elif model_name == "resnet50":
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)  # Load pretrained ResNet-50 model
        # Adjust the first convolutional layer for the number of input channels
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Adjust the final fully connected layer for the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)  # Forward pass

# Function to adjust the weights of the first convolutional layer
def adjust_conv1_weights(state_dict, input_channels):
    if input_channels == 1 and state_dict['model.conv1.weight'].shape[1] == 3:  # Check if the saved model has 3 input channels
        print("Adjusting conv1 weights for grayscale images...")
        weight = state_dict['model.conv1.weight']
        # Average the weights to convert them to a single channel
        state_dict['model.conv1.weight'] = weight.mean(dim=1, keepdim=True)

# Load the models and adjust weights if necessary
models = []
for path, model_name, input_channels in model_paths:
    model = PneumoniaModel(model_name, input_channels=input_channels)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    adjust_conv1_weights(state_dict, input_channels)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    models.append(model)

# Define the function to make ensemble predictions
def predict_ensemble(models, image, transform):
    outputs = []
    with torch.no_grad():  # Disable gradient tracking during inference
        for model in models:
            img = transform(image)  # Apply the transformation to the image
            if model.model.conv1.in_channels == 3:
                img = img.repeat(3, 1, 1)  # Convert single-channel image to 3 channels by repeating
            img = img.unsqueeze(0)  # Add batch dimension
            output = model(img)  # Get model predictions
            outputs.append(output)

    # Aggregate predictions using averaging
    outputs = torch.stack(outputs)
    avg_output = torch.mean(outputs, dim=0)  # Average outputs
    _, predicted = torch.max(avg_output, 1)  # Get the final predicted label
    return predicted.item()

# Load the test dataset
test_folder = os.path.join(location, 'test_images/test_images')
# Get the list of test images
test_images = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

# Make predictions on test images
predictions = []
file_names = []
print("Start making predictions...")
for file_name in test_images:
    image_path = os.path.join(test_folder, file_name)
    image = Image.open(image_path).convert('L')  # Ensure the image is in grayscale mode
    predicted_label = predict_ensemble(models, image, transform)  # Get the predicted label
    predictions.append(predicted_label)
    file_names.append(file_name)

# Create a DataFrame with file names and predicted labels
results_df = pd.DataFrame({'file_name': file_names, 'class_id': predictions})

# Save the DataFrame to a CSV file
results_path = os.path.join(location, 'results/v6.1_ensemble_best3.csv')
os.makedirs(os.path.dirname(results_path), exist_ok=True)  # Ensure the directory exists
results_df.to_csv(results_path, index=False)  # Save the predictions to a CSV file

print("Predictions saved to", results_path)