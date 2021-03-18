import os
import pdb
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from dataset import VisualLauguageBuilder
from model import VisualModel, VisionLanguageModel


VISION_ONLY_CKPT = '/mnt/fs1/ziyxiang/classes/PSYCH209FinalProject/checkpoints/vision-only_90.pth'
VISION_LANGUAGE_CKPT = '/mnt/fs1/ziyxiang/classes/PSYCH209FinalProject/checkpoints/vision-language_90.pth'


def load_args():
    parser = argparse.ArgumentParser(description='Running vision/vision-language model')
    parser.add_argument('--which_model', type=str,
                        help='Specify if using vision-only or vision-language model',
                        choices=['vision-only', 'vision-language'],
                        required=True)
    parser.add_argument('--embd_path', type=str,
                        help='Path to saved representations, if none save it during experiments')
    parser.add_argument('--result_folder', type=str,
                        help='Where to save trained models.')
    args = parser.parse_args()
    return args


class RunExp:
    def __init__(self, which_model, embd_path=None, result_folder=None):
        self.which_model = which_model
        self.embd_path = embd_path
        self.result_folder = result_folder
        if self.embd_path is None:    # only need to get embeddings when no saved embeddings are available
            self.__load_data()
            self.__load_model()
        self.load_label_map()
        self.load_embds()
        
    def __load_data(self):
        dataset = VisualLauguageBuilder(vision_only=False, analysis=True)    # returns normalized image and verbal desc
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    def __load_ckpt(self):
        if self.which_model == 'vision-only':
            ckpt_path = VISION_ONLY_CKPT
        else:
            ckpt_path = VISION_LANGUAGE_CKPT
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['state_dict'])
        
    def __load_model(self):        
        if self.which_model == 'vision-only':            
            self.model = VisualModel()            
        elif self.which_model == 'vision-language':
            self.model = VisionLanguageModel()
        self.__load_ckpt()
        
        if self.which_model == 'vision-language':
            self.model = self.model.visual
        self.model.cuda()
        self.model.eval()

    def load_label_map(self):
        text_label_path = '/mnt/fs1/ziyxiang/classes/PSYCH209FinalProject/data/textlabels.txt'
        f = open(text_label_path, 'r')
        self.label_map = {
            int(l.split()[1]) : l.split()[0] for l in f
        }
        
    def save_embds(self):
        all_embds = {}
        self.embd_path = os.path.join(
            self.result_folder, f'{self.which_model}_embds.pkl'
        )
        for i, (image, verbal_desc) in enumerate(self.dataloader):            
            image = image.cuda()
            embd, pred = self.model.get_embds(image)
            embd = embd.squeeze(0).detach().cpu().numpy()
            label_idx = torch.argmax(verbal_desc[:28]).item()
            if label_idx in all_embds:
                all_embds[label_idx].append(embd)
            else:
                all_embds[label_idx] = [embd]
        pickle.dump(all_embds, open(self.embd_path, 'wb'))

    def load_embds(self):
        if self.embd_path is None:
            self.save_embds()            
        assert self.which_model in self.embd_path, \
            'Mismatch between model type and saved embeddings, please check your path'
        self.all_embds = pickle.load(open(self.embd_path, 'rb'))
        
    def get_mean_embds(self):
        self.mean_embds = {}
        for label in self.all_embds:
            class_ = self.label_map[label]
            class_embds = np.stack(self.all_embds[label])
            self.mean_embds[class_] = class_embds.mean(axis=0)
            
    def get_dendrogram(self):
        classes = list(self.mean_embds.keys())
        mean_embds = np.asarray(list(self.mean_embds.values()))
        linkage_mat = linkage(mean_embds, 'ward')
        plt.figure(figsize=(18, 10))
        dendrogram(
            linkage_mat,
            labels=classes,
            leaf_rotation=90,
            leaf_font_size=8
        )
        plt.savefig(
            f'/mnt/fs1/ziyxiang/classes/PSYCH209FinalProject/figs/{self.which_model}_dendrogram.png',
            dpi=300
        )
            
    def vis_learned_visual_embds(self):  
        self.get_mean_embds()
        self.get_dendrogram()

    def compute_knn_scores(self, n_neighbors):        
        embds = []
        labels = []
        for label in self.all_embds:
            label_embds = self.all_embds[label]
            embds.extend(label_embds)
            labels.extend([label] * len(label_embds))            
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(embds, labels)
        scores = knn.score(embds, labels)
        return scores
        
        
if __name__ == '__main__':
    args = load_args()
    run_exp = RunExp(
        args.which_model, args.embd_path, args.result_folder
    )
    #run_exp.vis_learned_visual_embds()
