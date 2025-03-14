import os
import torch
import base64
import replicate

import numpy as np
from glob import glob
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.utils import make_grid
import torchvision.datasets as datasets
from IPython.display import Image, display
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"




# Neuronales Netzwerk basierter Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16):  # Initialisierung, alles hier drinnen passiert nur einmal
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(  # Representiert das Bild
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim) 
        )
        
        self.decoder = nn.Sequential(  # Erzuegt aus der Represenation wieder ein Bild
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Output between 0-1
        )

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):  # Zusammenbauen des Modells (Forward Pass)
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed.view(-1, 1, 28, 28)

    def train(self, _trainAutoencoder):
        # Definiert das Trainingsverhalten des Modells
        torch.device(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        model_path = "autoencoder.pth"
        if _trainAutoencoder:
            print("Training Autoencoder...")
            epochs = 10
            for epoch in range(epochs):
                for images, _ in self.dataloader:
                    images = images.to(self.device)
                    reconstructed = self(images)
                    loss = criterion(reconstructed, images)
        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
            # Speichern des trainierten Modells
            torch.save(self.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        else:
            # Wenn das Modell bereits trainiert wurde, soll dieses geladen werden
            if os.path.exists(model_path):
                self.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            else:
                print("No trained model found. Please train first.")

    def generateImages(self, creativitaet):
        # Laden der Daten
        real_images, _ = next(iter(self.dataloader))
        real_images = real_images.to(self.device)
        
        # Encodieren der Bilder
        with torch.no_grad():
            real_latents = self.encoder(real_images)  # Latente Representation 
        
        # Hier ändern wir unsere Representationen etwas ab
        noise = torch.randn_like(real_latents) * creativitaet  # Kleine Änderung
        new_latents = real_latents + noise  # Hinzufügen der kleinen Änderung
        
        # Neue Bilder, basierend auf den leicht veränderten Representationen
        with torch.no_grad():
            generated_images = self.decoder(new_latents)
        
        # Hier wandeln wir die Bilder von einem Vektor wieder in ein Array um
        generated_images = generated_images.view(-1, 28, 28)
        return generated_images


    def datenLaden(self):
        # Wir laden die Daten vom Ordner /data
        dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return dataset

    def displayImages(self, images):
        # Hier stellen wir die Daten dar
        fig, axes = plt.subplots(1, 8, figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
        plt.show()


# Faltungs Neuronaler Netzwerk Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()   # Initialisierung, alles hier drinnen passiert nur einmal

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (28,28) -> (28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (28,28) -> (14,14)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (14,14) -> (14,14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (14,14) -> (7,7)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (7,7) -> (14,14)
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (14,14) -> (28,28)
            nn.Sigmoid()  # Output between 0-1
        )

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def generateImages(self, creativitaet):
        real_images, _ = next(iter(self.dataloader))
        real_images = real_images.to(self.device)
        
        # Encode real images to get structured latent representations
        with torch.no_grad():
            real_latents = self.encoder(real_images)  # Get latent space representation
        
        # Add slight noise to real latents to create variations
        noise = torch.randn_like(real_latents) * 0.1  # Small perturbation
        new_latents = real_latents + noise  # Perturb latent vectors slightly
        
        # Generate new images by decoding perturbed latents
        with torch.no_grad():
            generated_images = self.decoder(new_latents)
        
        # Hier wandeln wir die Bilder von einem Vektor wieder in ein Array um
        generated_images = generated_images.view(-1, 28, 28)
        return generated_images

    def datenLaden(self):
        # Wir laden die Daten vom Ordner /data
        dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return dataset

    def train(self, _trainFaltungsAutoencoder):
        torch.device(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        
        model_path = "convautoencoder.pth"
        if _trainFaltungsAutoencoder:
            print("Training des Faltungs Autoencoder...")
            epochs = 10
            for epoch in range(epochs):
                for images, _ in self.dataloader:
                    images = images.to(self.device)
                    reconstructed = self(images)
                    loss = criterion(reconstructed, images)
        
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
            # Save the trained model
            torch.save(self.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        else:
            # Load trained model if exists
            if os.path.exists(model_path):
                self.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            else:
                print("No trained model found. Please train first.")

    def displayImages(self, images):
        # Hier stellen wir die Daten dar
        fig, axes = plt.subplots(1, 8, figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
        plt.show()

    def interpolate(self, steps):
        # Choose two digits (you can select specific indices from the dataset)
        real_images, _ = next(iter(self.dataloader))
        real_images = real_images.to(self.device)
        
        # Choose two random images from the batch (let's say image 0 and image 1 for example)
        image_1 = real_images[0].unsqueeze(0)  # First image
        image_2 = real_images[1].unsqueeze(0)  # Second image
        
        # Encode the two images to get their latent representations
        with torch.no_grad():
            latent_1 = self.encoder(image_1)  # Latent representation for image_1
            latent_2 = self.encoder(image_2)  # Latent representation for image_2
        
        # Interpolate between the two latent representations (latent vectors)
        interpolated_latents = []
        
        for alpha in torch.linspace(0, 1, steps):
            # Interpolate between latent_1 and latent_2
            latent_interpolated = (1 - alpha) * latent_1 + alpha * latent_2
            interpolated_latents.append(latent_interpolated)
        
        # Decode the interpolated latent vectors to generate images
        with torch.no_grad():
            generated_images = [self.decoder(latent) for latent in interpolated_latents]
        
        # Plot the generated images (interpolated between the two digits)
        fig, axes = plt.subplots(1, steps, figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(generated_images[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
        plt.show()


# VAE
class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()

        # Encoder: Conv + Pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # (28,28) -> (28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (28,28) -> (14,14)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # (14,14) -> (14,14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # (14,14) -> (7,7)
        )
        
        # Latent space: mean and log-variance
        self.fc_mu = nn.Linear(32 * 7 * 7, 128)  # Latent space mean
        self.fc_logvar = nn.Linear(32 * 7 * 7, 128)  # Latent space log variance

        # Decoder: ConvTranspose (Upsampling)
        self.decoder_input = nn.Linear(128, 32 * 7 * 7)  # Convert latent vector to shape (32, 7, 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (7,7) -> (14,14)
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (14,14) -> (28,28)
            nn.Sigmoid()  # Output between 0-1
        )

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 32, 7, 7)  # Reshape to (batch_size, 32, 7, 7)
        x_reconstructed = self.decoder(x)
        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

    def loss_function(self, x, x_reconstructed, mu, logvar):
        # Reconstruction loss (binary cross entropy)
        BCE = F.binary_cross_entropy(x_reconstructed, x, reduction='sum')

        # KL divergence
        # KL divergence between the learned distribution and the standard normal distribution
        # The formula is: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # where mu is the mean and sigma^2 is the variance
        # We use logvar (log of variance) directly, so we compute the KL divergence accordingly
        # The output of the encoder is already in the form of mu and logvar
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss is the sum of reconstruction loss and KL divergence
        return BCE + KL

    def datenLaden(self):
        # Wir laden die Daten vom Ordner /data
        dataset = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return dataset

    def train(self, _trainFaltungsVariationalAutoencoder):
        model = self.to(self.device)  # Replace ConvAutoencoder with ConvVAE
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model_path = "convvae.pth"
        
        # Training the VAE
        if _trainFaltungsVariationalAutoencoder:  # Update this flag if necessary
            print("Training the Variational Autoencoder...")
            epochs = 500
            for epoch in range(epochs):
                total_loss = 0
                for images, _ in self.dataloader:  # Assuming your dataloader is defined
                    images = images.to(self.device)
                    # Forward pass
                    reconstructed, mu, logvar = self(images)
                    # Compute VAE loss (reconstruction + KL divergence)
                    loss = self.loss_function(images, reconstructed, mu, logvar)
        
                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
                    total_loss += loss.item()
        
                avg_loss = total_loss / len(self.dataloader)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
            # Save the trained model
            torch.save(self.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        else:
            # Load trained model if exists
            if os.path.exists(model_path):
                self.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            else:
                print("No trained model found. Please train first.")

    def generateImages(self, num_samples):
        latent_dim = 128  # Latent space dimensionality (should match the latent vector size)
        
        # Sample random latent vectors from a normal distribution
        with torch.no_grad():  # No need to track gradients
            # Create random latent vectors from a normal distribution (mean=0, std=1)
            z = torch.randn(num_samples, latent_dim).to(self.device)
        
            # Decode the random latent vectors to generate images
            generated_images = self.decode(z)
        return generated_images

    def displayImages(self, images):
        # Plot the generated images
        fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
        for i, ax in enumerate(axes):
            ax.imshow(images[i].cpu().squeeze(), cmap="gray")
            ax.axis("off")
        plt.show()
        

class StyleGan():
    def __init__(self):
        # Laden des vortrainierten Modells (StyleGAN2) von PyTorch Hub
        self.model = torch.hub.load('ndahlquist/pytorch-hub-stylegan:0.0.1', 'style_gan', pretrained=True)
        self.model.eval()  # Set the model to evaluation mode

    def generateImages(self, num_samples):
        # Generate random latent vectors for face generation
        latent_vector = torch.randn(num_samples, 512)  # StyleGAN2 uses a latent space of size 512
        
        # Generate images from the latent vector
        with torch.no_grad():
            generated_images = self.model(latent_vector)
        return generated_images

    
    def displayImages(self, images):
        # Normalisieren der Bilder (-127-128 --> 0-255)
        images = (images + 1) / 2
        
        # Darstellen der erzeugten Bilder
        fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
        for i, ax in enumerate(axes):
            # Umwandeln der Bilder in ein Grid
            grid = make_grid(images[i], nrow=1)
            
            # Und darstellen dieser Grids
            ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
            ax.axis("off")
        plt.show()

    def interpolate(self, steps):
        # Zufällige Initialisierung für unsere Bilderstellung
        latent_vector_1 = torch.randn(1, 512)  # StyleGAN2 latenter Raum für Bild 1
        latent_vector_2 = torch.randn(1, 512)    # StyleGAN2 latenter Raum für Bild 2
        
        # Erstellen unseres Bildes 1 basierend auf dem Latenten Raum für Bild 1
        with torch.no_grad():
            image_1 = self.model(latent_vector_1).detach()
        
        # Erstellen unseres Bildes 1 basierend auf dem Latenten Raum für Bild 2
        with torch.no_grad():
            image_2 = self.model(latent_vector_2).detach()

        # Interpolieren in den Latenten Räumen und Erstellen der Bilder
        images = [image_1]
        for i in range(steps):
            c_factor = (i + 1) / (steps + 1)
            c_mixed_latent_vector = latent_vector_1.clone()
            c_mixed_latent_vector = (1 - c_factor) * latent_vector_1 + c_factor * latent_vector_2
            c_mixed_image = self.model(c_mixed_latent_vector).detach()

            images.append(c_mixed_image)
        images.append(image_2)
        return torch.cat(images, dim=0)


# Hilfssfunktionen um die LLama Familie darzustellen
def mm(graph):
  graphbytes = graph.encode("ascii")
  base64_bytes = base64.b64encode(graphbytes)
  base64_string = base64_bytes.decode("ascii")
  display(Image(url="https://mermaid.ink/img/" + base64_string))

def genai_app_arch():
  mm("""
  flowchart TD
    A[Users] --> B(Applications e.g. mobile, web)
    B --> |Hosted API|C(Platforms e.g. Custom, HuggingFace, Replicate)
    B -- optional --> E(Frameworks e.g. LangChain)
    C-->|User Input|D[Llama 3]
    D-->|Model Output|C
    E --> C
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def rag_arch():
  mm("""
  flowchart TD
    A[User Prompts] --> B(Frameworks e.g. LangChain)
    B <--> |Database, Docs, XLS|C[fa:fa-database External Data]
    B -->|API|D[Llama 3]
    classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def llama2_family():
  mm("""
  graph LR;
      llama-2 --> llama-2-7b
      llama-2 --> llama-2-13b
      llama-2 --> llama-2-70b
      llama-2-7b --> llama-2-7b-chat
      llama-2-13b --> llama-2-13b-chat
      llama-2-70b --> llama-2-70b-chat
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def llama3_family():
  mm("""
  graph LR;
      llama-3 --> llama-3-8b
      llama-3 --> llama-3-70b
      llama-3-8b --> llama-3-8b
      llama-3-8b --> llama-3-8b-instruct
      llama-3-70b --> llama-3-70b
      llama-3-70b --> llama-3-70b-instruct
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def llama3_1_family():
  mm("""
  graph LR;
      llama-3-1 --> llama-3-8b
      llama-3-1 --> llama-3-70b
      llama-3-1 --> llama-3-4050b
      llama-3-1-8b --> llama-3-1-8b
      llama-3-1-8b --> llama-3-1-8b-instruct
      llama-3-1-70b --> llama-3-1-70b
      llama-3-1-70b --> llama-3-1-70b-instruct
      llama-3-1-405b --> llama-3-1-405b-instruct
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

import ipywidgets as widgets
from IPython.display import display, Markdown

# Create a text widget
API_KEY = widgets.Password(
    value='',
    placeholder='',
    description='API_KEY:',
    disabled=False
)

def md(t):
  display(Markdown(t))

def bot_arch():
  mm("""
  graph LR;
  user --> prompt
  prompt --> i_safety
  i_safety --> context
  context --> Llama_3
  Llama_3 --> output
  output --> o_safety
  i_safety --> memory
  o_safety --> memory
  memory --> context
  o_safety --> user
  classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def fine_tuned_arch():
  mm("""
  graph LR;
      Custom_Dataset --> Pre-trained_Llama
      Pre-trained_Llama --> Fine-tuned_Llama
      Fine-tuned_Llama --> RLHF
      RLHF --> |Loss:Cross-Entropy|Fine-tuned_Llama
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def load_data_faiss_arch():
  mm("""
  graph LR;
      documents --> textsplitter
      textsplitter --> embeddings
      embeddings --> vectorstore
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

def mem_context():
  mm("""
      graph LR
      context(text)
      user_prompt --> context
      instruction --> context
      examples --> context
      memory --> context
      context --> tokenizer
      tokenizer --> embeddings
      embeddings --> LLM
      classDef default fill:#CCE6FF,stroke:#84BCF5,textColor:#1C2B33,fontFamily:trebuchet ms;
  """)

# Helper functions to run the different Llama models on Replicate
def llama2_7b(prompt):
    output = replicate.run(
      "meta/llama-2-7b-chat",
      input={"prompt": prompt}
    )
    return ''.join(output)

def llama2_70b(prompt):
    output = replicate.run(
      "meta/llama-2-70b-chat",
      input={"prompt": prompt}
    )
    return ''.join(output)

def llama3_8b(prompt):
    output = replicate.run(
      "meta/meta-llama-3-8b-instruct",
      input={"prompt": prompt}
    )
    return ''.join(output)

def llama3_70b(prompt):
    output = replicate.run(
      "meta/meta-llama-3-70b-instruct",
      input={"prompt": prompt}
    )
    return ''.join(output)


from PIL import Image
class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(128, 128)):
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        self.target_size = target_size  # Set target size for resizing

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize for faster training
            transforms.ToTensor(),  # Convert to tensor and normalize
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Resize both image and mask to the same target size
        img = img.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = torch.tensor(np.array(mask), dtype=torch.long) - 1 # Convert mask to tensor

        return img, mask


# Visualizing first few images and masks
def visualize_images_and_masks(dataset, num_samples=3):
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 4))

    for i in range(num_samples):
        img, mask = dataset[i]
        
        # Convert tensor image to numpy for plotting
        img_np = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        mask_np = mask.numpy()  # Convert mask tensor to numpy
        
        # Print the unique values in the mask
        print(f"Unique values in mask {i+1}: {np.unique(mask_np)}")

        # Plot image and mask
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f"Mask {i+1}")

        # Hide axes for both subplots
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def visualize_image_and_mask(img, mask):
    fig, axes = plt.subplots(1, 2, figsize=(8, 1 * 4))


    # Convert tensor image to numpy for plotting
    img_np = img.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
    mask_np = mask.numpy()  # Convert mask tensor to numpy

    # Plot image and mask
    axes[0].imshow(img_np)
    axes[0].set_title(f"Image")
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title(f"Mask")

    # Hide axes for both subplots
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


class UNet(nn.Module):
    def __init__(self, device, num_classes=3):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Encoder (Contracting Path)
        self.encoder1 = conv_block(3, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        # Middle
        self.middle = conv_block(256, 512)

        # Decoder (Expanding Path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        # Final output layer with the number of classes
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # num_classes = 3 for your task

        self.device = device

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        m = self.middle(self.pool(e3))

        d3 = self.decoder3(torch.cat([self.upconv3(m), e3], dim=1))
        d2 = self.decoder2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.decoder1(torch.cat([self.upconv1(d2), e1], dim=1))

        # Final convolution to get the output with 3 channels (for 3 classes)
        return self.final_conv(d1)

    def generateSegmentations(self, path):
        # Define the transformations (resize, normalize)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize to match input size of model
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
        ])
        
        # Load the image (assuming you have the cat image saved as 'cat.jpg')
        # path = "cat.jpg"
        # path = "Bengal_191.jpg"
        img = Image.open(path).convert("RGB")  # Open image and convert to RGB
        
        # Apply transformations
        img_tensor = transform(img).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Forward pass: Make a prediction
        with torch.no_grad():  # Disable gradients for inference
            output = self(img_tensor)  # Model output will be logits for each class
        
        # Apply softmax to get probabilities and then argmax to get the predicted class
        pred = torch.argmax(output, dim=1)  # Shape: (batch_size, 1, height, width)
        
        # Convert the prediction to a NumPy array for visualization
        pred_mask = pred.squeeze().cpu().numpy()  # Remove batch and channel dimensions
        
        # Plot the original image and the predicted mask
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis("off")
        
        # Predicted mask (segmentation)
        plt.subplot(1, 2, 2)
        plt.imshow(pred_mask, cmap="gray")  # Grayscale for the mask
        plt.title("Predicted Segmentation Mask")
        plt.axis("off")
        
        plt.show()



class UNetTrainer:
    def __init__(self, model, train_loader, device="cpu", model_name="unet_model.pth"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()  # Multi-class segmentation loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_model=True):
        if train_model:
            print("Training U-Net Model...")
            num_epochs = 25

            # Training loop
            for epoch in range(num_epochs):
                self.model.train()
                running_loss = 0.0

                # Use tqdm to add progress bar to the training loop
                for images, masks in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
                    images, masks = images.to(self.device), masks.to(self.device)

                    # Ensure masks are in LongTensor type for CrossEntropyLoss
                    masks = masks.long()

                    self.optimizer.zero_grad()

                    # Forward pass: get model outputs
                    outputs = self.model(images)

                    try:
                        # Compute the loss
                        loss = self.criterion(outputs, masks)
                    except Exception as e:
                        print(f"Error during loss calculation: {e}")
                        continue  # Skip this batch if there's an error

                    # Backward pass and optimizer step
                    try:
                        loss.backward()
                        self.optimizer.step()
                    except Exception as e:
                        print(f"Error during optimizer step: {e}")
                        continue  # Skip this batch if there's an error

                    running_loss += loss.item()

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(self.train_loader):.4f}")

            # Save the trained model
            torch.save(self.model.state_dict(), self.model_name)
            print(f"Model saved to {self.model_name}")
        
        else:
            # Load the model if it's already trained
            if os.path.exists(self.model_name):
                self.model.load_state_dict(torch.load(self.model_name, map_location=self.device))
                print(f"Model loaded from {self.model_name}")
            else:
                print("No trained model found. Please train the model first.")

