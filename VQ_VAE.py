import torch
import torch.nn as nn
from Encoder import residual_block, Encoder
from Decoder import Decoder
    
class VQuantized(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta):
        super(VQuantized, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        '''
            Args:
                x: (B, C, H, W)
            Returns:
                quantized: (B, C, H, W)
        '''
        x = x.permute(0, 2, 3, 1).contiguous() # B, H, W, dim
        batch_size, height, width, channel = x.shape
        x = x.view(-1, channel)

        # Calculate the distance between x and the codebook
        codebook = self.codebook.weight # num, dim
        distance = torch.sum(x**2, dim=1, keepdim=True) + torch.sum(codebook**2, dim=1) - 2 * torch.matmul(x, codebook.t())

        # Find the nearest codebook
        indices = torch.argmin(distance, dim=1).unsqueeze(1) # B*H*W
        quantized = self.codebook(indices).view(batch_size, height, width, channel)
        quantized = quantized.permute(0, 3, 1, 2).contiguous() # B, C, H, W
        return indices, quantized
    
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_resblocks, num_embeddings, beta):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim, num_resblocks)
        self.quantized = VQuantized(num_embeddings, embedding_dim, beta)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim, num_resblocks)

    def forward(self, x):
        x = self.encoder(x)
        indices, quantized = self.quantized(x)
        x_recon = self.decoder(quantized)
        return x_recon

if __name__ == '__main__':
    model = VQuantized(512, 64, 0.25)
    input = torch.randn(1, 64, 32, 32)
    indices, quantized = model(input)
    print(indices.shape, quantized.shape)
    # print(indices)
    # print(quantized)

