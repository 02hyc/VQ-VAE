import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Datasets import CelebADataset, transform
from VQ_VAE import VQVAE
from tqdm import tqdm
import wandb
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(
    project="VQVAE-Training",
    config={
        "learning_rate": 2.25e-4,
        "batch_size": 64,
        "num_epochs": 10,
        "hidden_dim": 256,
        "embedding_dim": 64,
        "num_embeddings": 512,
        "beta": 0.25,
    }
)

IMG_PATH = '/home/yukino/Documents/Code/Datasets/CelebA/Img/img_align_celeba'
BATCH_SIZE = 64
num_epochs = 10

dataset = CelebADataset(IMG_PATH, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

vqvae_model = VQVAE(input_dim=3, hidden_dim=256, embedding_dim=128, num_resblocks=4, num_embeddings=512).to(device)

def custom_loss_function(input, output, ori_quantized, encoder_output):

    reconstruction_loss = torch.mean((output - input) ** 2)
    
    quantized_loss = torch.mean((ori_quantized.detach() - encoder_output) ** 2)
    
    commitment_loss = torch.mean((encoder_output.detach() - ori_quantized) ** 2)
    
    loss = reconstruction_loss + quantized_loss + 0.25 * commitment_loss

    return loss

optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=2.25e-4)

best_loss = float('inf')
for epoch in range(num_epochs):
    vqvae_model.train()
    with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} - Training") as pbar:
        for data in pbar:
            optimizer.zero_grad()
            
            data = data.to(device)
            encoder_output, quantized, ori_quantized, output = vqvae_model(data)
            
            loss = custom_loss_function(data, output, ori_quantized, encoder_output)
            reconstruction_loss = torch.mean((output - data) ** 2)
            quantized_loss = torch.mean((ori_quantized.detach() - encoder_output) ** 2)
            commitment_loss = torch.mean((encoder_output.detach() - quantized) ** 2)
            
            loss.backward()
            optimizer.step()
            
            wandb.log({
                "train_loss": loss.item(),
                "train_reconstruction_loss": reconstruction_loss.item(),
                "train_quantized_loss": quantized_loss.item(),
                "train_commitment_loss": commitment_loss.item(),
            })
            
            pbar.set_postfix(loss=loss.item())
            
            if pbar.n == len(train_dataloader) - 1:
                with torch.no_grad():
                    sample_size = min(8, data.size(0))
                    sample_images = data[:sample_size]
                    sample_reconstructions = output[:sample_size]
                    
                    sample_images = (sample_images + 1) / 2
                    sample_reconstructions = (sample_reconstructions + 1) / 2
                    
                    image_grid = torchvision.utils.make_grid(
                        torch.cat([sample_images, sample_reconstructions], dim=0),
                        nrow=sample_size
                    )
                    
                    wandb.log({
                        "reconstructions": wandb.Image(image_grid, caption=f"Epoch {epoch+1} Reconstructions")
                    })
    
    vqvae_model.eval()
    val_loss = 0
    val_reconstruction_loss = 0
    val_quantized_loss = 0
    val_commitment_loss = 0
    num_batches = len(val_dataloader)
    
    with torch.no_grad():
        for data in val_dataloader:
            data = data.to(device)
            encoder_output, quantized, ori_quantized, output = vqvae_model(data)
            
            loss = custom_loss_function(data, output, quantized, encoder_output)
            reconstruction_loss = torch.mean((output - data) ** 2)
            quantized_loss = torch.mean((ori_quantized.detach() - encoder_output) ** 2)
            commitment_loss = torch.mean((encoder_output.detach() - ori_quantized) ** 2)
            
            val_loss += loss.item()
            val_reconstruction_loss += reconstruction_loss.item()
            val_quantized_loss += quantized_loss.item()
            val_commitment_loss += commitment_loss.item()
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(vqvae_model.state_dict(), f"model_best.pt")
        
    val_loss /= num_batches
    val_reconstruction_loss /= num_batches
    val_quantized_loss /= num_batches
    val_commitment_loss /= num_batches
    
    wandb.log({
        "val_loss": val_loss,
        "val_reconstruction_loss": val_reconstruction_loss,
        "val_quantized_loss": val_quantized_loss,
        "val_commitment_loss": val_commitment_loss,
        "epoch": epoch + 1
    })
    
    print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")