import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Datasets import CelebADataset, transform
from VQ_VAE import VQVAE


IMG_PATH = '/home/yukino/Documents/Code/Datasets/CelebA/Img/img_align_celeba'
BATCH_SIZE = 64
num_epochs = 10

dataset = CelebADataset(IMG_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vqvae_model = VQVAE(3, 256, 64, 4, 512, 0.25)

def custom_loss_function(input, output, quantized, encoder_output):

    reconstruction_loss = torch.mean((output - input) ** 2)
    
    quantized_loss = torch.mean((quantized.detach() - encoder_output) ** 2)
    
    commitment_loss = torch.mean((encoder_output.detach() - quantized) ** 2)
    
    loss = reconstruction_loss + quantized_loss + commitment_loss

    return loss

optimizer = torch.optim.Adam(vqvae_model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        
        output, quantized, encoder_output = vqvae_model(data)
        
        loss = custom_loss_function(data, output, quantized, encoder_output)
        
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')