import numpy as np
import cv2
import os

class Dataloader:
    def __init__(self,data_path,batch_size,shuffle=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = 0
        self.current_index = 0
        self.train_input_set = []
        self.train_output_set = []
        self.valid_input_set = []
        self.valid_output_set = []
    def load_data(self):
        total_path = os.walk(self.data_path)
        for path,dir_list,file_list in total_path:
            for file_name in file_list:
                if "instance" in file_name or "part" in file_name or "json" in file_name:
                    continue
                else:
                    im = cv2.imread(os.path.join(path,file_name)).transpose((2,0,1))
                    if im.shape[1] < 256 or im.shape[2] < 256:
                        continue
                    im = im[:,0:256, 0:256]
                    image = np.array(im)
                    if 'seg' in file_name:
                        if 'train' in file_name:
                            self.train_output_set.append(image)
                        else:
                            self.valid_output_set.append(image)
                    else:
                        if 'train' in file_name:
                            self.train_input_set.append(image)
                        else:
                            self.valid_input_set.append(image)

    #def preprocess_data(self):
    #    pass
    #def iterate_batches(self):
    #    pass

a = Dataloader(r"C:\Users\DELL\Desktop\test",1)
a.load_data()
#print(len(a.valid_input_set))
#print(len(a.valid_output_set))
#print(len(a.train_input_set))
#print(len(a.train_output_set))
#print(a.valid_input_set[0].shape)









#im = cv2.imread(r'C:\Users\DELL\Desktop\ADE20K_2021_17_01\images\ADE\training\unclassified\canal_urban\ADE_frame_00000078.jpg')
##im.resize(2000,2000,3)
#im = im[0:256,0:256]
#image = np.array(im)
#print(image.shape)

#print(im.shape)
#imageio.imwrite('filename.jpg', image)







