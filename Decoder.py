import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import residual_block

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_resblocks):
        super(Decoder, self).__init__()
        self.resblock = nn.Sequential(
            *[residual_block(embedding_dim, embedding_dim, 3, 1, 1) for _ in range(num_resblocks)]
        )
        self.batch_norm1 = nn.BatchNorm2d(embedding_dim)
        self.deconv1 = nn.ConvTranspose2d(embedding_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(hidden_dim)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        for resblock in self.resblock:
            x = self.batch_norm1(x)
            x = resblock(x)
        # print("resblock: ", x.shape)
        x = self.deconv1(x)
        # print("deconv1: ", x.shape)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        # print("deconv2: ", x.shape)
        x = F.tanh(x)
        return x
    
if __name__ == '__main__':
    model = Decoder(64, 256, 3, 4)
    input = torch.randn(1, 64, 32, 32)
    output = model(input)
    print(output.shape)

