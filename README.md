# Pneumonia Detection Using Deep Learning

## 1. Introduction

This project focuses on detecting pneumonia from chest X-ray images using deep learning techniques. The images are grayscale and belong to one of three classes:
- Class 0: No disease (healthy)
- Class 1: Bacterial pneumonia
- Class 2: Viral pneumonia

The dataset is derived from the publicly available **Chest X-Ray Images (Pneumonia)** dataset. The task involves building models to classify the images into the correct categories based on their features, and the performance is evaluated using accuracy, defined as:

**Accuracy** = (Number of correctly predicted test images) / (Total number of test images)

## 2. Data

The data is divided into two main folders:
- `train_images` contains 4,672 images, split as follows:
  - 1,227 images for class 0
  - 2,238 images for class 1
  - 1,207 images for class 2
- `test_images` contains 1,168 images for evaluation.

Additionally, a `labels_train.csv` file provides the corresponding labels for the training images.

## 3. Experiments

### 3.1 Architectures

We experimented with several CNN architectures, including **ResNet-18**, **ResNet-34**, and **ResNet-50**, to build models capable of accurately classifying chest X-ray images. Due to the relatively small dataset size, smaller and less complex architectures, like ResNet-18 and ResNet-34, performed better than more complex models.

### 3.2 Data Augmentation

To improve model generalization and prevent overfitting, we employed various data augmentation techniques such as:
- Rotation
- Shear transformations
- Scaling
- Zooming
- Adjusting brightness and contrast

These augmentations helped in diversifying the dataset and improved the model's ability to generalize to unseen test data.

### 3.3 Automatic Image Generation

In addition to traditional augmentation techniques, we employed **automatic image generation** to address the class imbalance issue and expand the dataset size. This was particularly important because the dataset had a limited number of samples, which could hinder the performance of deep learning models.

For each class, we generated up to **50,000** images by creating synthetic variations of existing X-rays using a custom augmentation function. This function dynamically applied transformations such as rotation, width/height shifts, shearing, zooming, and brightness adjustments. The augmentation was applied until the desired number of images per class was achieved. By doing this, we ensured that the model had sufficient data for each class to learn effectively.

This technique allowed us to increase the dataset size significantly, leading to better model generalization and performance.

### 3.4 Learning Rate Scheduling

We explored two learning rate schedulers:
- **ReduceLROnPlateau**: Decreases the learning rate when the validation accuracy plateaus, allowing finer adjustments during training.
- **CyclicLR**: Helps the model escape local minima by cyclically increasing and decreasing the learning rate.

### 3.5 Ensemble Models

We created ensemble models combining different ResNet architectures to lower variance and leverage the strengths of different models. The ensemble models consistently outperformed individual models in terms of accuracy.

## 4. Results

The results of the experiments indicate that the ensemble models performed better than the individual models in detecting pneumonia from chest X-ray images. Below is the table summarizing the performance of various models based on test accuracy:

| **Model Name** | **Architecture(s) Used**            | **Accuracy (%)** |
|----------------|-------------------------------------|------------------|
| V6EB3          | ResNet-18 (V3R18), ResNet-50 (V3R50), ResNet-50 (V7R50) | 86.472           |
| V5EB2          | ResNet-18 (V3R18), ResNet-50 (V3R50) | 86.301           |
| V7EB3          | ResNet-18 (V3R18), ResNet-50 (V3R50), ResNet-50 (V1R50) | 86.130           |
| V2EB4          | ResNet-18 (V3R18), ResNet-50 (V3R50), ResNet-50 (V7R50), ResNet-50 (V15R50) | 85.787           |
| V1R50          | ResNet-50 | 85.445           |
| V3R18          | ResNet-18 | 84.589           |
| V3R50          | ResNet-50 | 84.246           |
| V7R50          | ResNet-50 | 83.732           |
| V15R50         | ResNet-50 | 83.732           |
| V4R50          | ResNet-50 | 83.39            |

### Individual Model Details

| **Model Name** | **Samples per Class (\*1000)** | **Batch Size** | **Learning Rate** | **Epochs Trained** | **Accuracy (%)** |
|----------------|-------------------------------|----------------|-------------------|--------------------|------------------|
| V1R50          | 30                            | 9              | 0.0001            | 7                  | 85.445           |
| V3R18          | 30                            | 32             | 0.001 CLR         | 14                 | 84.589           |
| V3R50          | 50                            | 60             | 0.0001 RLRP       | 53                 | 84.246           |
| V7R50          | 30                            | 120            | 0.0001            | 10                 | 83.732           |
| V15R50         | 30                            | 32             | 0.0001 CLR        | 24                 | 83.732           |
| V4R50          | 50                            | 9              | 0.0001 RLRP       | 9                  | 83.39            |

The highest accuracy of **86.472%** was achieved by the **V6EB3** ensemble model, which combined ResNet-18 and ResNet-50 models. The **V5EB2** ensemble model came close with an accuracy of **86.301%**. Individual models like **V1R50** (ResNet-50) achieved **85.445%**, while **V3R18** (ResNet-18) achieved **84.589%**, demonstrating that smaller models can perform competitively with proper hyperparameter tuning.

## 5. Conclusions

Our experiments showed that ensemble models significantly outperformed individual models in pneumonia detection from chest X-rays. Specifically, models combining ResNet-18 and ResNet-50 architectures yielded the highest accuracies. ResNet-18 demonstrated strong performance, especially with learning rate schedulers like ReduceLROnPlateau.

Data augmentation and learning rate scheduling played a critical role in improving model performance. In future work, we plan to further enhance the models by exploring new hyperparameters, expanding the dataset, and potentially using transfer learning techniques. This project demonstrates the potential of deep learning in automating medical diagnostics for pneumonia detection.
