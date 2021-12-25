import torch 
import torch.nn as nn 
#########################################
#    1 x 1 conv CAE
#########################################
class FeatCAE(nn.Module):
    def __init__(self, in_channels=3456, latent_dim=200, is_bn=True):
        super(FeatCAE, self).__init__()
        self.Autoencoder = torch.nn.Sequential()
        self.Autoencoder.add_module("conv_1", torch.nn.Conv2d(in_channels, (in_channels + latent_dim)//2, kernel_size=1, stride=1, padding=0))
        if is_bn:
            self.Autoencoder.add_module("bn_1", nn.BatchNorm2d(num_features=(in_channels + latent_dim) // 2))
        self.Autoencoder.add_module("relu_1", nn.ReLU())
        self.Autoencoder.add_module("conv_2", torch.nn.Conv2d((in_channels + latent_dim)//2, 2*latent_dim, kernel_size=1, stride=1, padding=0))
        if is_bn:
            self.Autoencoder.add_module("bn_2", nn.BatchNorm2d(num_features=2*latent_dim))
        self.Autoencoder.add_module("relu_2", nn.ReLU())
        self.Autoencoder.add_module("conv_3", torch.nn.Conv2d(2*latent_dim, latent_dim, kernel_size=1, stride=1, padding=0))
        self.Autoencoder.add_module("conv_4", torch.nn.Conv2d(latent_dim, 2*latent_dim, kernel_size=1, stride=1, padding=0))
        if is_bn:
            self.Autoencoder.add_module("bn_3", nn.BatchNorm2d(num_features=2*latent_dim))
        self.Autoencoder.add_module("relu_3", nn.ReLU())    
        self.Autoencoder.add_module("conv_5", torch.nn.Conv2d(2*latent_dim,  (in_channels + latent_dim)//2, kernel_size=1, stride=1, padding=0))
        if is_bn:
            self.Autoencoder.add_module("bn_4", nn.BatchNorm2d(num_features= (in_channels + latent_dim)//2))
        self.Autoencoder.add_module("relu_4", nn.ReLU())   
        self.Autoencoder.add_module("conv_6", torch.nn.Conv2d((in_channels + latent_dim)//2,  in_channels, kernel_size=1, stride=1, padding=0)) 
    def forward(self, x):
        x = self.Autoencoder(x)    
        return x

if __name__ == "__main__":    
    input = torch.randn(1, 3456, 56, 56)
    model = FeatCAE( in_channels=3456, latent_dim=200, is_bn=True)
    print(model(input).shape)    
    