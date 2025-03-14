import torch
import torch.nn as nn
import torch.nn.functional as F

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_resblocks):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, embedding_dim, kernel_size=4, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(embedding_dim)
        self.resblock = nn.Sequential(
            *[residual_block(embedding_dim, embedding_dim, 3, 1, 1) for _ in range(num_resblocks)]
        )


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.batch_norm1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.batch_norm2(x))
        for i, resblock in enumerate(self.resblock):
            x = resblock(x)
            if i < len(self.resblock) - 1:
                x = self.batch_norm2(x)
        return x
    
if __name__ == '__main__':
    model = Encoder(3, 256, 64, 4)
    input = torch.randn(1, 3, 128, 128)
    output = model(input)
    print(output.shape)
    # print(model)