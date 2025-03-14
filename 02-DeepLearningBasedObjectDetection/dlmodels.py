import requests
import skimage
import torch
import clip
import cv2
import os

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import models
from PIL import Image

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class VGG16ObjectDetector:
    # Alles was in __init__ steht, machen wir nur einmal
    def __init__(self):
        # Laden des so genannten VGG16 Modells
        self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.model.eval()  # Modell wird nicht trainiert, sondern verwendet
        
        # Übersetzen der Modell Ausgabe in lesbaren Text (Tasse, Computer, Hund, Katze)
        LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        self.labels = np.array(requests.get(LABELS_URL).text.splitlines()) 
        
        # Bilder müssen für das Modell in eine bestimmte Form gebracht werden. Das machen wir hier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Richtige Bilddimensionen 224x224 (VGG Größe)
            transforms.ToTensor(),  # Umwandeln in etwas, das vom Modell gelesen werden kann
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisierung
        ])

    # Alles was hier steht wird jedesmal ausgeführt
    def predict(self, frame, top5=False):
        frame_pil = Image.fromarray(frame)
        input_tensor = self.transform(frame_pil).unsqueeze(0)
    
        # Hier wenden wir die Filter auf das Bild an und verwenden diese Filter um das, was im Bild gefunden wurde, zu erkennen 
        with torch.no_grad():
            output = self.model(input_tensor)
    
        if not top5:
            # Hier geben wir das ähnlichste Objekt aus
            top1_probs, top1_indices = torch.topk(output, 1)  # hier bekommen wir das ähnlichste Objekt
            top1_probs = torch.nn.functional.softmax(top1_probs, dim=1)[0]  # Und wandeln dieses in Warscheindlichkeiten um
            top1_labels = [self.labels[idx] for idx in top1_indices[0].tolist()]  # Und geben es in eine Liste (damit wir immer eine Liste ausgeben)
            return top1_labels
    
        # Hier geben wir die 5 ähnlichsten Objekt aus
        top5_probs, top5_indices = torch.topk(output, 5)  # hier bekommen wir die 5 ähnlichsten
        top5_probs = torch.nn.functional.softmax(top5_probs, dim=1)[0]  # Und wandeln diese in Warscheindlichkeiten um
        top5_labels = [self.labels[idx] for idx in top5_indices[0].tolist()]  # Und geben diese in eine Liste
        return top5_labels

class RestNet50ObjectDetector:
    # Alles was in __init__ steht, machen wir nur einmal
    def __init__(self):
        # Laden des so genannten Resnet50 Modells
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()  # Modell wird nicht trainiert, sondern verwendet
        
        # Übersetzen der Modell Ausgabe in lesbaren Text (Tasse, Computer, Hund, Katze)
        LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        self.labels = np.array(requests.get(LABELS_URL).text.splitlines())
        
        # Bilder müssen für das Modell in eine bestimmte Form gebracht werden. Das machen wir hier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Richtige Bilddimensionen 224x224 (VGG Größe)
            transforms.ToTensor(),  # Umwandeln in etwas, das vom Modell gelesen werden kann
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisierung
        ])

    def predict(self, frame, top5=False):
        frame_pil = Image.fromarray(frame)
        input_tensor = self.transform(frame_pil).unsqueeze(0)  # Add batch dimension
    
        # Anwenden des Modells auf unsere Daten um die Objekte zu erkennen
        with torch.no_grad():
            output = self.model(input_tensor)
    
        if not top5:
            # Hier geben wir das ähnlichste Objekt aus
            top1_probs, top1_indices = torch.topk(output, 1)  # hier bekommen wir das ähnlichste Objekt
            top1_probs = torch.nn.functional.softmax(top1_probs, dim=1)[0]  # Und wandeln dieses in Warscheindlichkeiten um
            top1_labels = [self.labels[idx] for idx in top1_indices[0].tolist()]  # Und geben es in eine Liste (damit wir immer eine Liste ausgeben)
            return top1_labels
    
        # Hier geben wir die 5 ähnlichsten Objekt aus
        top5_probs, top5_indices = torch.topk(output, 5)  # hier bekommen wir die 5 ähnlichsten
        top5_probs = torch.nn.functional.softmax(top5_probs, dim=1)[0]  # Und wandeln diese in Warscheindlichkeiten um
        top5_labels = [self.labels[idx] for idx in top5_indices[0].tolist()]  # Und geben diese in eine Liste
        return top5_labels

class RestNet152ObjectDetector:
    # Alles was in __init__ steht, machen wir nur einmal
    def __init__(self):
        # Laden des so genannten Resnet50 Modells
        self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.model.eval()  # Modell wird nicht trainiert, sondern verwendet
        
        # Übersetzen der Modell Ausgabe in lesbaren Text (Tasse, Computer, Hund, Katze)
        LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        self.labels = np.array(requests.get(LABELS_URL).text.splitlines())
        
        # Bilder müssen für das Modell in eine bestimmte Form gebracht werden. Das machen wir hier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Richtige Bilddimensionen 224x224 (VGG Größe)
            transforms.ToTensor(),  # Umwandeln in etwas, das vom Modell gelesen werden kann
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisierung
        ])

    def predict(self, frame, top5=False):
        frame_pil = Image.fromarray(frame)
        input_tensor = self.transform(frame_pil).unsqueeze(0)  # Add batch dimension
    
        # Anwenden des Modells auf unsere Daten um die Objekte zu erkennen
        with torch.no_grad():
            output = self.model(input_tensor)
    
        if not top5:
            # Hier geben wir das ähnlichste Objekt aus
            top1_probs, top1_indices = torch.topk(output, 1)  # hier bekommen wir das ähnlichste Objekt
            top1_probs = torch.nn.functional.softmax(top1_probs, dim=1)[0]  # Und wandeln dieses in Warscheindlichkeiten um
            top1_labels = [self.labels[idx] for idx in top1_indices[0].tolist()]  # Und geben es in eine Liste (damit wir immer eine Liste ausgeben)
            return top1_labels
    
        # Hier geben wir die 5 ähnlichsten Objekt aus
        top5_probs, top5_indices = torch.topk(output, 5)  # hier bekommen wir die 5 ähnlichsten
        top5_probs = torch.nn.functional.softmax(top5_probs, dim=1)[0]  # Und wandeln diese in Warscheindlichkeiten um
        top5_labels = [self.labels[idx] for idx in top5_indices[0].tolist()]  # Und geben diese in eine Liste
        return top5_labels

class MaskRCNN:
    # Alles was in __init__ steht, machen wir nur einmal
    def __init__(self):
        # Laden des so genannten FastRCNN Modells. Für die Objekterkennung verwendet MaskRCNN wiederum ResNet-50.
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()  # Modell wird nicht trainiert, sondern verwendet

    def predict(self, frame):
        # Um die Objekte zu erkennen und deren Position zu finden muss zuerst die Farbe des Bildes von BGR zu RGB übersetzt werden
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Danach bringen wir das Bild in eine Form, welche für unser Modell lesbar ist ...
        image_tensor = torch.tensor(image_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
        input_tensor = [image_tensor]
    
        # Und verwenden dann unser Modell um Objekte und deren Position herauszufinden
        with torch.no_grad():
            predictions = self.model(input_tensor)
    
        # Vorhersagen der Position der Objekte, der Objekte selbst und der Warscheindlichkeit
        pred_boxes = predictions[0]["boxes"].cpu().numpy()
        pred_labels = predictions[0]["labels"].cpu().numpy()
        pred_scores = predictions[0]["scores"].cpu().numpy()

        return pred_boxes, pred_labels, pred_scores

class ClipObjectAndTextEmbedding:
    # Alles was in __init__ steht, machen wir nur einmal
    def __init__(self, device):
        print(f'Mögliche Clip Embeddings: {clip.available_models()}')
        print('Wir wählen: "ViT-B/32"')

        # Wir laden das Modell
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)
        # Und setzten es auf den Ausführmodus (wir trainieren es nicht weiter)
        self.model.eval()
        self.device = device
        
        input_resolution = self.model.visual.input_resolution
        context_length = self.model.context_length
        vocab_size = self.model.vocab_size
        print(f'Bild Größe: {input_resolution}')
        print(f'Text Context Länge: {context_length}')
        print(f'Bekannte Wörter des Modells: {vocab_size}')

    def tokenize(self, text):
        return clip.tokenize(text)

    def loadImages(self, descriptions):
        # Hier laden wir die Bilder
        bilder, originalBilder = [], []
        for filename in [filename for filename in os.listdir(skimage.data_dir) if filename.endswith('.png') or filename.endswith('.jpg')]:
            name = os.path.splitext(filename)[0]

            if name not in descriptions:
                continue
            
            image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")
            originalBilder.append(image)
            bilder.append(self.preprocess(image))
        return bilder, originalBilder

    def showImages(self, bilder, beschreibungen):
        # Hier stellen wir die Bilder dar
        plt.figure(figsize=(16, 5))
        for counter, bild in enumerate(bilder):
            plt.subplot(2, 4, counter+1)
            plt.imshow(bild)
            plt.title(f"{beschreibungen[counter]}")
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()

    def plotSimilarity(self, originalBilder, beschreibungen, similarity):
        # Hier geben wir die Ähnlichkeit von Bildern und Texten aus
        count = len(beschreibungen)

        plt.figure(figsize=(20, 14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        # plt.colorbar()
        plt.yticks(range(count), beschreibungen, fontsize=18)
        plt.xticks([])
        for i, image in enumerate(originalBilder):
            plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y, x]:.2f}", ha="center", va="center", size=12)
        
        for side in ["left", "top", "right", "bottom"]:
          plt.gca().spines[side].set_visible(False)
        
        plt.xlim([-0.5, count - 0.5])
        plt.ylim([count + 0.5, -2])
        
        plt.title("Cosine similarity between text and image features", size=20)
        plt.show()

    def processImage(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        return self.preprocess(frame_pil).unsqueeze(0).to(self.device)

    def calculateSimilarity(self, bilder, beschreibungen):
        # Hier berechnen wir die Ähnlichkeit zwischen Texte und Bilder
        # Erst wandeln wir die Bilder und Texte in eine Nummerform um
        image_input = torch.tensor(np.stack(bilder)).to(self.device)
        text_tokens = self.tokenize(["This is " + desc for desc in beschreibungen]).to(self.device)
        with torch.no_grad():
            # Hier berechnen wir die Bild und Text CLIP representation
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Hier berechnen wir die Ähnlichkeit zwischen der Text und Bild CLIP Representation
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        return similarity
