import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        self.conv_encoder_c64 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_encoder_c128 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_encoder_c256 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_encoder_c512_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_encoder_c512_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_encoder_c512_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_encoder_c512_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Encoder
        self.encoder = nn.Sequential(
            self.conv_encoder_c64,
            self.conv_encoder_c128,
            self.conv_encoder_c256,
            self.conv_encoder_c512_1,
            self.conv_encoder_c512_2,
            self.conv_encoder_c512_3,
            self.conv_encoder_c512_4
        )

        self.conv_decoder_c512_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder_c1024_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder_c1024_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder_c1024_3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder_c512_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder_c256 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_decoder_c128 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


        # Decoder
        self.decoder = nn.Sequential(
            self.conv_decoder_c512_1,
            self.conv_decoder_c1024_1,
            self.conv_decoder_c1024_2,
            self.conv_decoder_c1024_3,
            self.conv_decoder_c512_2,
            self.conv_decoder_c256,
            self.conv_decoder_c128
        )
        self.conv1_layer = nn.ConvTranspose2d(64, out_channels, kernel_size=1)
        self.tanh_layer = nn.Tanh()

    def forward(self, x):
        # Encoder
        enc_outputs = []
        for module in self.encoder:
            x = module(x)
            enc_outputs.append(x)
        # Decoder
        for module in self.decoder:
            x = torch.cat((x, enc_outputs.pop()), dim=1)
            x = module(x)
        x = self.conv1_layer(x)
        x = self.tanh_layer(x)
        return x




