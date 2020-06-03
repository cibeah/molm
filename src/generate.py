import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets

from generator import Generator
from scores import InceptionScore, FID, Identity
from train import load_data


logger = logging.Logger("generate")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=str,
                    help="path to the .pt checkpoint file")
parser.add_argument("--dataset", type=str, default="../data/CIFAR",
                    help="path to the training data. Defauls to ../data/CIFAR")
parser.add_argument("--device", type=str, default = "cuda:0",
                    help="name of the device to be used for training (only one device cause we aren't rich)")
parser.add_argument("--savepath", "-s", type=str, default=os.getcwd(),
                    help="path to a folder to save the true and generated images")

def sample_true(dataset, batch_size=64):
    """
    Returns a samble r
    """
    loader = torch.utils.data.DataLoader(dataset,
                                        shuffle=True,
                                        batch_size=batch_size)
    samples, _ = iter(loader).next()
    return samples

def generate(G, batch_size=64):
    z =  torch.randn(batch_size, G.dims[0], device=device)
    generated = G(z).detach().cpu()
    generated = (generated +1) / 2
    return generated

def save_grid(samples, path):
    grid = torchvision.utils.make_grid(samples, nrow=6) # 6 images per row
    plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.savefig(path)

if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    device_name = args.device
    savepath = args.savepath
    
    #Sample true data and save
    dataset = load_data(args.dataset)
    samples = sample_true(dataset)
    save_grid(samples, savepath + "/true_img.png")
    logger.info("True samples images saved at {}".format(savepath + "/true_img.png"))

    #Generate samples from checkpoint and save
    device = torch.device(device_name)
    G = Generator().to(device)
    checkpoint = torch.load(checkpoint_path)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    generated = generate(G)
    save_grid(generated, savepath + "/generated.png")
    logger.info("Generated images saved at {}".format(savepath + "/generated.png"))
