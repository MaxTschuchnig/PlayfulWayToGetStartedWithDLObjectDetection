import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# ==== DEFINE SMALL CNN ====
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 32)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        

# ==== DEFINE REGULARIZED SMALL CNN ====
class SmallRegCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallRegCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 32 * 32, 32)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add dropout with 50% probability

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after first FC layer
        x = self.fc2(x)
        return x


# ==== DEFINE CNN ====
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        

# ==== DEFINE REGULARIZED CNN ====
class CNNReg(nn.Module):
    def __init__(self, num_classes):
        super(CNNReg, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input images are 128x128
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.75)  # Add dropout with 50% probability

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# ==== DEFINE REGULARIZED BN CNN ====
class CNNRegBN(nn.Module):
    def __init__(self, num_classes):
        super(CNNRegBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Assuming input images are 128x128
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 32)
        self.bn_fc2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)  # Add dropout with 50% probability

        # Bilder müssen für das Modell in eine bestimmte Form gebracht werden. Das machen wir hier
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize all images to 128x128
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalization
        ])

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def predict(self, frame, DEVICE, labels, top5=False):
        frame_pil = Image.fromarray(frame)
        input_tensor = self.transform(frame_pil).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE) 
    
        # Hier wenden wir die Filter auf das Bild an und verwenden diese Filter um das, was im Bild gefunden wurde, zu erkennen 
        with torch.no_grad():
            output = self(input_tensor)
    
        if not top5:
            # Hier geben wir das ähnlichste Objekt aus
            top1_probs, top1_indices = torch.topk(output, 1)  # hier bekommen wir das ähnlichste Objekt
            top1_probs = torch.nn.functional.softmax(top1_probs, dim=1)[0]  # Und wandeln dieses in Warscheindlichkeiten um
            top1_labels = [labels[idx] for idx in top1_indices[0].tolist()]  # Und geben es in eine Liste (damit wir immer eine Liste ausgeben)
            return top1_labels
    
        # Hier geben wir die 5 ähnlichsten Objekt aus
        top5_probs, top5_indices = torch.topk(output, 5)  # hier bekommen wir die 5 ähnlichsten
        top5_probs = torch.nn.functional.softmax(top5_probs, dim=1)[0]  # Und wandeln diese in Warscheindlichkeiten um
        top5_labels = [labels[idx] for idx in top5_indices[0].tolist()]  # Und geben diese in eine Liste
        return top5_labels

