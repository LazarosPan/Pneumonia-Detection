import os
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torchvision.models import resnet50, ResNet50_Weights
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, LearningRateMonitor
import torch
from PIL import Image
import multiprocessing

torch.set_float32_matmul_precision('high')  # Set higher precision for matrix multiplications

num_cpu_cores = multiprocessing.cpu_count()  # Get the number of CPU cores

# Read the .csv file containing labels and file names
location = "detect-pneumonia-spring-2024/"  # Adjust the path accordingly

df = pd.read_csv(location + "labels_train.csv")

# Get the count of images in each class
class_counts = df['class_id'].value_counts()

# Define the target number of images per class
target_images_per_class = 30000

# Calculate the number of additional images needed for each class
additional_images_needed = target_images_per_class - class_counts
#print(additional_images_needed)

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the augmentation function using OpenCV
def augment_image(image):
    # Random rotation
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Random width and height shift
    tx = np.random.uniform(-0.1, 0.1) * image.shape[1]
    ty = np.random.uniform(-0.1, 0.1) * image.shape[0]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Random shear
    shear_factor = np.random.uniform(-0.2, 0.2)
    M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Random zoom
    zx, zy = np.random.uniform(0.9, 1.1, 2)
    image = cv2.resize(image, None, fx=zx, fy=zy, interpolation=cv2.INTER_LINEAR)

    # Random horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    return image

# Define the custom dataset
class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_images_per_class=target_images_per_class):
        self.annotations = pd.read_csv(csv_file)  # Read the annotations from the CSV file
        self.root_dir = root_dir  # Directory with all the images
        self.transform = transform  # Transformation to be applied on images
        self.target_images_per_class = target_images_per_class  # Target number of images per class
        
        # Calculate the number of images needed for each class
        self.class_counts = self.annotations['class_id'].value_counts()
        self.additional_images_needed = target_images_per_class - self.class_counts

        # Create lists of images per class
        self.image_paths_by_class = {class_id: [] for class_id in self.additional_images_needed.index}
        for idx, row in self.annotations.iterrows():
            self.image_paths_by_class[row['class_id']].append(row['file_name'])
        
    def __len__(self):
        return self.target_images_per_class * len(self.additional_images_needed)  # Total number of images in the dataset

    def __getitem__(self, idx):
        # Determine the class and the specific image index within that class
        class_id = idx // self.target_images_per_class
        img_idx = idx % self.target_images_per_class
        
        if img_idx < self.class_counts[class_id]:
            # Original image
            img_name = self.image_paths_by_class[class_id][img_idx]
            img_path = os.path.join(self.root_dir, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Augmented image
            img_name = np.random.choice(self.image_paths_by_class[class_id])
            img_path = os.path.join(self.root_dir, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = augment_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        label = class_id  # Assign the class ID as the label
        
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert image to PIL Image
    transforms.Resize((224, 224)),  # Resize the image
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
])

# Create the dataset and dataloader
dataset = PneumoniaDataset(
    csv_file=location + 'labels_train.csv',
    root_dir=location + 'train_images/train_images',
    transform=transform,
    target_images_per_class=target_images_per_class
)

dataloader = DataLoader(dataset, batch_size=120, shuffle=True, num_workers=num_cpu_cores, pin_memory=True, persistent_workers=True)

# Define the model class
class PneumoniaModel(pl.LightningModule):
    def __init__(self, num_classes=3):
        super(PneumoniaModel, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)  # Load pretrained ResNet-50 model
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adjust the first convolution layer for grayscale input
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # Adjust the final fully connected layer for the number of classes

    def forward(self, x):
        return self.model(x)  # Forward pass

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)  # Get model predictions
        loss = F.cross_entropy(outputs, labels)  # Compute cross-entropy loss
        self.log('train_loss', loss)  # Log the training loss
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)  # Use Adam optimizer with specified learning rate

# Define callbacks for early stopping and learning rate monitoring
early_stopping_callback = EarlyStopping(monitor='train_loss', patience=3, verbose=True, mode='min')
lr_monitor = LearningRateMonitor(logging_interval='epoch')

if __name__ == '__main__':
    # Instantiate the model
    model = PneumoniaModel()
    #model.to(device)
    cudnn.benchmark = True  # Enable cuDNN auto-tuner for faster training

    # Set up the progress bar callback
    progress_bar = TQDMProgressBar(refresh_rate=10)

    # Define the Trainer with ProgressBar enabled
    trainer = pl.Trainer(max_epochs=50, accelerator='cuda', callbacks=[progress_bar, early_stopping_callback, lr_monitor]) if torch.cuda.is_available() else pl.Trainer(max_epochs=10, callbacks=[progress_bar, early_stopping_callback, lr_monitor])

    # Start training
    trainer.fit(model, dataloader)

    # Define the transform to preprocess the images for prediction
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
    ])

    # Path to the folder containing testing images
    test_folder = location + 'test_images/test_images'

    # Define a function to predict the class for a single image
    def predict_image(model, image_path, transform):
        image = Image.open(image_path).convert('L')  # Ensure the image is in grayscale mode
        image = transform(image).unsqueeze(0)  # Add batch dimension
        #image = image.to(device)  # Move the image to the same device as the model
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient tracking during inference
            outputs = model(image)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predicted label
        return predicted.item()

    # Iterate over testing images and make predictions
    predictions = []
    file_names = []
    for file_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, file_name)
        predicted_label = predict_image(model, image_path, transform)  # Predict the class of the image
        predictions.append(predicted_label)
        file_names.append(file_name)

    # Create a DataFrame with file names and predicted labels
    results_df = pd.DataFrame({'file_name': file_names, 'class_id': predictions})

    # Save the DataFrame to a CSV file
    results_df.to_csv(location + 'results/v7_resnet50_gpu_s30k_b120_e50_p3.csv', index=False)

    # Specify the directory and file name to save the model
    model_save_path = location + "models/v7_resnet50_gpu_s30k_b120_e50_p3.pth"

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)