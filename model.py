import torch.nn as nn
import torch
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from compressai.models.base import CompressionModel

class Network(CompressionModel):
    def __init__(self, N=128):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(N)
        self.encode = nn.Sequential(
            nn.Conv2d(3, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
            GDN(N),
            nn.Conv2d(N, N, stride=2, kernel_size=5, padding=2),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, padding=2, output_padding=1, stride=2),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, padding=2, output_padding=1, stride=2)
        )

    def forward(self, x):
        y = self.encode(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decode(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }
    
    def compress(self, x):
        y = self.encode(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}
    
    def decompress(self, y_strings, shape):
        y_hat = self.entropy_bottleneck.decompress(y_strings, shape)
        x_hat = self.decode(y_hat)
        return x_hat

if __name__ == '__main__':
    net = Network()
    x = torch.rand(1, 3, 256, 256)
    output = net(x)
    print(output["x_hat"].shape, output["likelihoods"]["y"].shape)
