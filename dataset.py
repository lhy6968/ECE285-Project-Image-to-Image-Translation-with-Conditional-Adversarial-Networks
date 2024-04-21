from handle_image import crop_random_subimage
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy
import numpy as np
class GanDataset(Dataset):
    def __init__(self,root_dir,data_type):
        self.root_dir = root_dir
        self.data_type = data_type
        self.dataset = []
        self.labels = []
        self.data_name_list = []
        self.label_name_list = []

    def __len__(self):
        return len(self.dataset)


    def load_data(self):
        total_path = os.walk(self.root_dir)
        for path,dir_list,file_list in total_path:
            for file_name in file_list:
                if "instance" in file_name or "part" in file_name or "json" in file_name:
                    continue
                elif self.data_type not in file_name:
                    continue
                else:
                    image_path = os.path.join(path, file_name)
                    image = Image.open(image_path)

                    # Get the width and height of the image
                    width, height = image.size
                    if width < 256 or height < 256:
                      continue
                    else:
                        #if len(image.shape) != 3:
                         #   continue
                        #image = image.transpose((2, 0, 1))
                        if "seg" in file_name:
                          self.data_name_list.append(os.path.join(path, file_name))
                        else:
                          self.label_name_list.append(os.path.join(path, file_name))
        self.data_name_list.sort()
        self.label_name_list.sort()
        print(self.data_name_list[0:5])
        print(self.label_name_list[0:5])
        print(self.data_name_list[-1])
        print(self.label_name_list[-1])
        print(self.data_name_list[-2])
        print(self.label_name_list[-2])
        print(len(self.dataset))
        print(len(self.labels))
        for i in range(len(self.data_name_list)):
          image = Image.open(self.data_name_list[i])

          # Get the width and height of the image
          width, height = image.size
          x = np.random.randint(0, width - 256 + 1)
          y = np.random.randint(0, height - 256 + 1)
          label_image = crop_random_subimage(self.label_name_list[i], 256, x, y)
          if len(label_image.shape) != 3:
            print(self.label_name_list[i])
            print(self.data_name_list[i])
            continue
          else:
            label_image = label_image.transpose((2, 0, 1)) 
          data_image = crop_random_subimage(self.data_name_list[i], 256, x, y).transpose((2, 0, 1))
          self.dataset.append(data_image)
          self.labels.append(label_image)
        print(len(self.dataset))
        print(len(self.labels))
          

    def __getitem__(self, idx):
        data = torch.from_numpy(self.dataset[idx])
        label = torch.from_numpy(self.labels[idx])
        return data,label


# a = GanDataset(r"C:\Users\DELL\Desktop\test","val")
# a.load_data()
# val_dataloader = DataLoader(a, batch_size=4, shuffle=True)
# print(len(a))
# for batch_idx, (real_input, target_output) in enumerate(val_dataloader):
#     tensor = target_output[0]
#     # Scale the tensor values to the range [0, 255]
#     #tensor = (tensor + 1) / 2 * 255  # Assuming the tensor values are in the range [-1, 1]
#     print(tensor.shape)
#     # Convert the tensor to a PIL Image
#     image = Image.fromarray(tensor.byte().permute(1, 2, 0).cpu().numpy(), mode='RGB')
#
#     # Save the image as a PNG file
#     image.save(r"C:\Users\DELL\Desktop\%d.png" % (2))
# #a.load_data()
# #print(len(a))
# #b,c = a[0]
# #print(b.shape)
# #print(c.shape)

#device = 'cuda' if torch.cuda.is_available() else 'cpu'

#a.to(device)




