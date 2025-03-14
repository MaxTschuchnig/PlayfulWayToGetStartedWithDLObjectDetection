import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from collections import Counter
import random
import time
import models

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ==== DATALOADERS ====
def get_dataloaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS):
    # Loading the data once, and reuse later

    # ==== DATA AUGMENTATION ====
    simple_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize all images to 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
    ])
    
    # ==== LOAD DATASET USING DataLoader FOR PARALLEL PROCESSING ====
    train_dataset = datasets.ImageFolder(root=DATA_DIR + "/train", transform=simple_transform)
    test_dataset = datasets.ImageFolder(root=DATA_DIR + "/test", transform=simple_transform)
    
    # Use DataLoader to parallelize loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # ==== PRELOAD DATA INTO RAM (Now Faster!) ====
    def preload_data(dataloader):
        images, labels = [], []
        for img, lbl in dataloader:
            images.append(img)
            labels.append(lbl)
        return torch.cat(images), torch.cat(labels)
    
    print("Loading training data into RAM...")
    train_images, train_labels = preload_data(train_loader)  # Parallel loading
    
    print("Loading testing data into RAM...")
    test_images, test_labels = preload_data(test_loader)  # Parallel loading
    
    # Save the class names BEFORE conversion
    class_labels = train_dataset.classes
    
    # Convert to TensorDataset
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    # Create final DataLoader (now using RAM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # No need for workers now
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print("Data is now fully loaded into RAM!")
    return train_loader, test_loader, class_labels


# ==== MODEL SAVING AND LOADING ====
def safe_model(model, path):
    torch.save(model.state_dict(), path)
    

def load_model(path, model_class, NUM_CLASSES, DEVICE):
    model = model_class(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(path))
    return model


def get_balanced_trainloader(DATA_DIR, BATCH_SIZE, NUM_WORKERS):
    # Loading the data once, and reuse later
    
    # ==== DATA AUGMENTATION ====
    simple_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize all images to 128x128
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
    ])
    
    # ==== LOAD DATASET USING DataLoader FOR PARALLEL PROCESSING ====
    train_dataset = datasets.ImageFolder(root=DATA_DIR + "/train", transform=simple_transform)
    
    # Use DataLoader to parallelize loading
    train_loader_balanced = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # ==== PRELOAD DATA INTO RAM (Now Faster!) ====
    def preload_data(dataloader):
        images, labels = [], []
        for img, lbl in dataloader:
            images.append(img)
            labels.append(lbl)
        return torch.cat(images), torch.cat(labels)
    
    print("Loading training data into RAM...")
    train_images_balanced, train_labels_balanced = preload_data(train_loader_balanced)  # Parallel loading
    
    # ==== FIND MINIMUM CLASS SIZE FOR UNDERSAMPLING ====
    class_counts = Counter(train_labels_balanced.numpy())  # Count occurrences of each label
    min_class_size = min(class_counts.values())  # Smallest class count
    
    print(f"Class distribution before undersampling: {class_counts}")
    print(f"Using {min_class_size} samples per class for balance.")
    
    # ==== UNDERSAMPLING: SELECT MIN_CLASS_SIZE SAMPLES PER CLASS ====
    selected_indices = []
    for class_label in class_counts:
        class_indices = (train_labels_balanced == class_label).nonzero(as_tuple=True)[0].tolist()
        selected_indices.extend(random.sample(class_indices, min_class_size))  # Randomly sample min_class_size
    
    # Create balanced dataset
    train_images_balanced = train_images_balanced[selected_indices]
    train_labels_balanced = train_labels_balanced[selected_indices]
    
    # Print new class distribution
    balanced_class_counts = Counter(train_labels_balanced.numpy())
    print(f"Class distribution after undersampling: {balanced_class_counts}")
    
    # ==== CONVERT TO FINAL DATASET & DATALOADER ====
    train_dataset_balanced = TensorDataset(train_images_balanced, train_labels_balanced)
    train_loader_balanced = DataLoader(train_dataset_balanced, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    print("Undersampling applied! Training data is now balanced.")
    return train_loader_balanced


# ==== LABEL SMOOTHING LOSS ====
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + (1 - one_hot) * self.smoothing / (n_class - 1)
        return torch.mean(torch.sum(-one_hot * F.log_softmax(pred, dim=1), dim=1))


# ==== CONFUSION MATRIX ====
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


# ==== TRAINING FUNCTION ====
def train(model, train_loader, criterion, optimizer, DEVICE):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    start_time = time.time()

    for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    return running_loss / len(train_loader), acc


# ==== TESTING FUNCTION ====
def test(model, test_loader, criterion, DEVICE):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Save predictions & labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100.0 * correct / total
    return running_loss / len(test_loader), acc, all_preds, all_labels


def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 6))
    
    # Losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Testing Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()

    # Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, test_accuracies, label='Testing Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()

    # Show plots
    plt.tight_layout()
    plt.show()
