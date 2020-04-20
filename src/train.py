import argparse
import json
import logging

import torch
import torchvision
import torchvision.transforms as transforms

from generator import Generator
from moment_network import MomentNetwork
from trainer import Trainer


logger = logging.Logger("train")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


parser = argparse.ArgumentParser()
parser.add_argument("params_path", type=str,
                    help="path to the .json parameters file")
parser.add_argument("--dataset", type=str, default="../data/CIFAR",
                    help="path to the training data. Defauls to ../data/CIFAR")
parser.add_argument("--device", type=str, default = "cuda:0",
                    help="name of the device to be used for training (only one device cause we aren't rich)")


def load_data(dataset):
    return torchvision.datasets.CIFAR10(
    root=dataset,
    train=True,
    download=True, #download if non existant at the location 'root'
    transform=transforms.Compose([
        transforms.ToTensor() #we want our data to be loaded as tensors
    ])
)

if __name__ == "__main__":
    args = parser.parse_args()
    params_path = args.params_path
    with open(params_path, "r") as f:
        params_dict = json.load(f)
    dataset = args.dataset
    device_name = args.device

    logger.info("\n Launching training run with parameters: \n {}, \n -- training dataset: {}, \
                \n -- device: {}, \n ".format(params_dict, dataset, device_name))


    train_set = load_data(dataset)

    device = torch.device(device_name)
    G = Generator().to(device)
    MoNet = MomentNetwork().to(device)
    trainer = Trainer(G, MoNet, train_set, params_dict, device, learn_moments=True)
    # trainer.generate_and_display(trainer.fixed_z, save=True, save_path=trainer.save_path + "generated_molm_cifar10_iter{}.png".format(1))
    trainer.train(save_images=True)

