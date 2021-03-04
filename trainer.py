import os
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm
from dataset import VisualLauguageBuilder
from model import VisualModel, VisionLanguageModel

def load_args():
    parser = argparse.ArgumentParser(description='Running vision/vision-language model')
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size')
    parser.add_argument('--init_lr', type=float,
                        help='Start learning rate')
    parser.add_argument('--save_freq', type=int,
                        help='Save checkpoint every N epoch')
    parser.add_argument('--which_model', type=str,
                        help='Specify if using vision-only or vision-language model',
                        choices=['vision-only', 'vision-language'],
                        required=True)
    parser.add_argument('--result_folder', type=str,
                        help='Where to save trained models.')
    args = parser.parse_args()
    return args

def save_checkpoint(model, epoch, optimizer, losses, args):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    ckpt_path = os.path.join(
        args.result_folder, f'{args.which_model}_{epoch}.pth')
    torch.save(checkpoint, ckpt_path)

def train(args):
    all_losses = []
    if args.which_model == 'vision-only':
        model = VisualModel()        
        dataset = VisualLauguageBuilder(vision_only=True)
    elif args.which_model == 'vision-language':
        model = VisionLanguageModel()
        dataset = VisualLauguageBuilder(vision_only=False)
    model.cuda()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    optimizer = Adam(model.parameters(), lr=args.init_lr)
    for epoch in tqdm(range(args.num_epochs)):
        epoch_losses = []
        for i, (item1, item2) in tqdm(enumerate(dataloader)):
            item1 = item1.cuda()
            item2 = item2.cuda()
            optimizer.zero_grad()
            loss = model(item1, item2)            
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        all_losses.extend(epoch_losses)
        if epoch % args.save_freq == 0:
            save_checkpoint(model, epoch, optimizer, all_losses, args)


if __name__ == '__main__':
    args = load_args()
    train(args)
    
