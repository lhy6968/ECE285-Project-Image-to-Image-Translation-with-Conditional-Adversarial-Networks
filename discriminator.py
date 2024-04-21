import torch.nn as nn
import torch
class Discriminator(nn.Module):
    def __init__(self, input_channels, num_filters=64):
        super(Discriminator, self).__init__()

        self.conv1_layer = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.relu_layer = nn.LeakyReLU(0.2, inplace=True)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2_layer = nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)

        self.sigmoid_layer = nn.Sigmoid()



    def forward(self, x,y):
        x = torch.cat((x, y), dim=1)
        x = self.conv1_layer(x)
        x = self.relu_layer(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv2_layer(x)
        x = self.sigmoid_layer(x)

        return x
# Create an instance of the discriminator
#input_channels = 6  # Number of channels in the input images
#discriminator = Discriminator(input_channels)

# Forward pass
#input_image = torch.randn(1, 3, 256, 256)  # Example input image
#output_image = torch.randn(1, 3, 256, 256)
#output = discriminator(input_image,output_image)

# Print the output shape
#print(output.shape)
#print(output)