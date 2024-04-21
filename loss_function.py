import torch.nn as nn
import torch

def adversarial_loss(discriminator,generator_output,real_images,real_output):
    criterion = nn.BCEWithLogitsLoss()
    real_output = discriminator(real_images, real_output)
    fake_output = discriminator(generator_output.detach(), real_output)

    real_labels = torch.ones_like(real_output)
    fake_labels = torch.zeros_like(fake_output)

    real_loss = criterion(real_output, real_labels)
    fake_loss = criterion(fake_output, fake_labels)
    discriminator_loss = (real_loss + fake_loss) * 0.5
    return discriminator_loss

def l1_loss(generator_output,target_images):
    criterion = nn.L1Loss()
    l1_loss = criterion(generator_output, target_images)
    return l1_loss