import os
import pdb
import pickle
import numpy as np
from PIL import Image

class DataPreprocessing(object):
    def __init__(self, raw_folder, processed_folder, raw_txt_labels):        
        self.num_classes = 28    # pre-set number in the dataset
        self.raw_folder = raw_folder
        self.processed_folder = processed_folder
        self.raw_txt_labels = raw_txt_labels
        self.meta = {}    # meta information to be saved, including class: {'verbal_descriptor': [], }
        self.load_raw_txt_labels()
        self.check_image_txt_classes()
        self.save_resized_images()
        
    def make_one_hot_name_vector(self, label_idx):
        name_vector = np.zeros(self.num_classes)
        name_vector[label_idx] = 1
        return name_vector
        
    def load_raw_txt_labels(self):
        self.classes = []
        f = open(self.raw_txt_labels, 'r')
        for line in f:
            raw_info = line.split()
            class_ = raw_info[0]            
            label_idx = int(raw_info[1])
            label = self.make_one_hot_name_vector(label_idx)
            descriptor = np.asarray(raw_info[3:], dtype=int)
            descriptor = np.append(label, descriptor)
            self.classes.append(class_)
            self.meta[class_] = {'verbal_descriptor': [], 'image_paths': []}
            self.meta[class_]['verbal_descriptor'] = descriptor
        
    def check_image_txt_classes(self):
        img_classes = os.listdir(self.raw_folder)
        assert set(img_classes) == set(self.classes), \
            "Image classes don't match verbal classes"

    def load_resize_one_image(self, im_raw_path, im_processed_path):
            im = Image.open(im_raw_path).convert('RGB')
            im = im.resize((256, 256))
            im.save(im_processed_path)

    def save_resized_images(self):        
        for class_ in self.classes:
            im_raw_folder = os.path.join(self.raw_folder, class_)
            im_processed_folder = os.path.join(
                self.processed_folder, class_)
            try:
                os.mkdir(im_processed_folder)
            except:
                print(f'{im_processed_folder} already exists')
                pass
            im_files = os.listdir(im_raw_folder)
            im_processed_paths = []
            for path in im_files:
                im_raw_path = os.path.join(im_raw_folder, path)
                im_processed_path = os.path.join(im_processed_folder, path)
                #self.load_resize_one_image(im_raw_path, im_processed_path)
                im_processed_paths.append(im_processed_path)
            self.meta[class_]['image_paths'] = im_processed_paths
        pickle.dump(self.meta, open('data/meta.pkl', 'wb'))
        
        
if __name__ == '__main__':
    raw_folder = 'data/raw_images/'
    processed_folder = 'data/processed_images/'
    raw_text_labels = 'data/textlabels.txt'
    dp = DataPreprocessing(raw_folder, processed_folder, raw_text_labels)
    dp.load_raw_txt_labels()
