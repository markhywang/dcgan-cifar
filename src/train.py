"""
Trains the Deep Convolutional Generative Adversarial Network on CIFAR-10.

The CIFAR-10 dataset can be found here: https://www.cs.toronto.edu/~kriz/cifar.html
"""

# Imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from model import Discriminator, Generator, init_weights

# Set random seed for reproducibility
torch.manual_seed(1337)

# Devices and other assets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
step_count = 0
MODEL_PATH = Path("../models")
MODEL_PATH.mkdir(
    parents=True, # create parent directories if needed
    exist_ok=True # if models directory already exists, don't error
)

# Hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
LATENT_DIM = 100
EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
ADAM_BETAS = (0.5, 0.999)


def train(gen: nn.Module, disc: nn.Module, dataloader: torch.utils.data.DataLoader,
          optim_gen: torch.optim, optim_disc: torch.optim, criterion: nn.Module) -> None:
    """
    Training step for DCGAN model.

    Note that, just like any other GAN, DCGANs operates on unsupervised learning.
    """
    global step_count
    
    for batch_idx, (real, _) in enumerate(dataloader):
        # Get real and fake images
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, LATENT_DIM, 1, 1).to(device)
        fake = gen(noise)
        
        # Train Discriminator: maximize log(D(real)) + log(1 - D(G(latent)))
        disc_real = disc(real).reshape(-1).to(device)
        disc_fake = disc(fake.detach()).reshape(-1).to(device)

        loss_real = criterion(disc_real, torch.ones_like(disc_real))
        loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_real + loss_fake) / 2 # Calculate mean loss
        
        disc.zero_grad()
        loss_disc.backward()
        optim_disc.step()
        
        # Train Generator: minimize log(1 - D(G(latent)) <-> maximize log(D(G(latent)))
        output = disc(fake).reshape(-1).to(device)
        loss_gen = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        loss_gen.backward()
        optim_gen.step()
        
        # Print losses occasionally to Tensorboard
        if batch_idx == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] \
                  Loss Discriminator: {loss_disc:.4f}, Loss Generator: {loss_gen:.4f}"
            )

            with torch.inference_mode():
                fake = gen(fixed_noise)

                # Take out at most 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step_count)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step_count)

            step_count += 1


def save_model(model: nn.Module, model_name: str) -> None:
    """
    Saves specifed model to a directory
    """
    save_path = MODEL_PATH / model_name
    print(f"Saving model to: {save_path}")
    torch.save(
        obj=model.state_dict(),
        f=save_path
    )


if __name__ == "__main__":
    # Prepare image transforms
    image_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            # Normalize with mean and standard deviation of 0.5 for proper scaling
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        )
    ])
    
    # Get dataset and dataloader
    dataset = datasets.CIFAR10(
        root="../data/",
        train=True,
        transform=image_transforms,
        download=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Create SummaryWriters for visualizations
    writer_real = SummaryWriter(f"../runs/real")
    writer_fake = SummaryWriter(f"../runs/fake")
    
    # Initialize discriminator and generator models, with weights
    discriminator = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    generator = Generator(LATENT_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)

    models_path = Path(__file__).resolve().parent.parent / "models"
    if models_path.exists(): # If model exists, then update state dict weights
        print("Base model exists. Updating state dict...")

        disc_state_dict = torch.load(f="../models/discriminator-model.pth", weights_only=False)
        discriminator.load_state_dict(disc_state_dict)
        
        gen_state_dict = torch.load(f="../models/generator-model.pth", weights_only=False)
        generator.load_state_dict(gen_state_dict)
    else: # Otherwise, initialize weights
        print("Initializing new model...")
        
        init_weights(discriminator)
        init_weights(generator)
    
    # Create fixed random noise of batch size 32 (for output display)
    fixed_noise = torch.randn(size=(32, LATENT_DIM, 1, 1)).to(device)
    
    # Create criterions and optimizer
    optim_gen = optim.Adam(
        params=generator.parameters(), 
        lr=LEARNING_RATE, 
        betas=ADAM_BETAS
    )
    optim_disc = optim.Adam(
        params=discriminator.parameters(), 
        lr=LEARNING_RATE, 
        betas=ADAM_BETAS
    )
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(EPOCHS):
        train(
            gen=generator,
            disc=discriminator,
            dataloader=dataloader,
            optim_gen=optim_gen,
            optim_disc=optim_disc,
            criterion=criterion
        )
    
    # Save both discriminator and generator models
    save_model(discriminator, "discriminator-model.pth")
    save_model(generator, "generator-model.pth")
