import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from discriminator import Discriminator
from generator import Generator
from tqdm import tqdm
from eval import evaluate
from save_model import handle_model
from PIL import Image
import random

def train_model(learning_rate,batch_size,num_epochs,train_dataset,val_dataset,save_model_or_not,load_model_or_not,eval_model_dict,load_gen_dict,load_dis_dict,save_gen_dict,save_dis_dict):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if load_model_or_not == True:
        generator = Generator(3, 3)
        discriminator = Discriminator(6)
        generator.load_state_dict(torch.load(load_gen_dict))
        discriminator.load_state_dict(torch.load(load_dis_dict))
        generator.to(device)
        discriminator.to(device)
        for param in generator.parameters():
          param.to(device)
        for param in discriminator.parameters():
          param.to(device)
    else:
        generator = Generator(3, 3)
        discriminator = Discriminator(6)
        generator.to(device)
        discriminator.to(device)
        for param in generator.parameters():
          param.to(device)
        for param in discriminator.parameters():
          param.to(device)
        torch.save(generator.state_dict(), save_gen_dict)
        torch.save(discriminator.state_dict(), save_dis_dict)

    lambda_task = 100

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    for epoch in tqdm(range(num_epochs)):
        total_discriminator_loss = 0.0
        total_generator_loss = 0.0
        for batch_idx, (real_input, target_output) in enumerate(train_dataloader):
            real_input = real_input.to(device).float()
            target_output = target_output.to(device).float()

            #real_input_array = real_input.numpy()
            #real_input_image = Image.fromarray(real_input_array.byte().permute(1, 2, 0).cpu().numpy(), mode='RGB')
            real_input /= 127.5
            real_input -= 1
            target_output /= 127.5
            target_output -= 1
            # Update discriminator weights
            discriminator_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            real_outputs = discriminator(real_input, target_output).to(device).float()
            real_labels = torch.ones_like(real_outputs).to(device).float()
            real_loss = criterion_gan(real_outputs, real_labels)

            fake_inputs = generator(real_input).to(device).float()
            fake_outputs = discriminator(fake_inputs.detach(), target_output).to(device).float()
            fake_labels = torch.zeros_like(fake_outputs).to(device).float()
            fake_loss = criterion_gan(fake_outputs, fake_labels)

            discriminator_loss = (real_loss + fake_loss) * 0.5
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Update generator weights 
            fake_outputs = discriminator(fake_inputs.detach(), target_output).to(device).float()
            generator_loss_gan = criterion_gan(fake_outputs, real_labels)
            #generator_loss_gan = criterion_gan(fake_inputs, target_output)
            generator_loss_l1 = criterion_l1(fake_inputs, target_output)
            generator_loss = (generator_loss_gan + lambda_task * generator_loss_l1)

            generator_loss.backward()
            generator_optimizer.step()

            total_discriminator_loss += discriminator_loss.item()
            total_generator_loss += generator_loss.item()

        avg_discriminator_loss = total_discriminator_loss / len(train_dataloader)
        avg_generator_loss = total_generator_loss / len(train_dataloader)
        # Print the losses
        print(f"Epoch [{epoch + 1}/{num_epochs}], Discriminator Loss: {avg_discriminator_loss:.4f}, Generator Loss: {avg_generator_loss:.4f}")

        if save_model_or_not == True:
            torch.save(generator.state_dict(), save_gen_dict)
            torch.save(discriminator.state_dict(), save_dis_dict)

        if epoch % 1 == 0:
            target_output_list, eval_output_list = evaluate(generator,val_dataloader)
            idx = random.randint(0, len(target_output_list))
            target_output = target_output_list[idx]
            eval_output = eval_output_list[idx]
            eval_output += 1
            eval_output *= 127.5
            print(eval_output.max())
            print(eval_output.min())
            print(eval_output)
            target_image = Image.fromarray(target_output.byte().permute(1, 2, 0).cpu().numpy(), mode='RGB')
            eval_image = Image.fromarray(eval_output.byte().permute(1, 2, 0).cpu().numpy(), mode='RGB')
            target_path = eval_model_dict + ("/%d_target.png")%(idx)
            eval_path = eval_model_dict + ("/%d_eval.png")%(idx)
            target_image.save(target_path)
            eval_image.save(eval_path)
