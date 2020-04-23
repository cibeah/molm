import argparse
import copy
import json
import logging
import pickle

import torch
import torchvision
import torchvision.transforms as transforms

from generator import Generator
from moment_network import MomentNetwork
from trainer import Trainer
from scores import InceptionScore, FID, Identity


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
parser.add_argument("scores", type=str, nargs='*', default=["FID", "IS"],
                    help="scores used to evaluate the model")
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

def load_scores(scores, device):
    inception_v3 = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=True)
    inception_v3.eval()
    scores_dict = {}
    if "FID" in scores:
        logger.info("FID will be used for scoring")
        model_fid = FID.make_fid_model(inception_v3)
        # logger.info("Fitting FID")
        with open("scoring/fid.pickle", "rb") as f:
            #loading directly to avoid computing the feature on the whole dataset again
            scores_dict["FID"] = pickle.load(f)
    if "IS" in scores:
        logger.info("Inception Score will be used for scoring")
        scores_dict["IS"] = InceptionScore(inception_v3, device)
    return scores_dict

if __name__ == "__main__":
    args = parser.parse_args()
    params_path = args.params_path
    with open(params_path, "r") as f:
        params_dict = json.load(f)
    dataset = args.dataset
    device_name = args.device
    scores = args.scores
 
    logger.info("\n Launching training run with parameters: \n {}, \n -- training dataset: {}, \
                \n -- device: {}, \n -- scores: {}".format(params_dict, dataset, device_name, scores))


    train_set = load_data(dataset)

    device = torch.device(device_name)
    scores_dict = load_scores(scores, device)
    G = Generator().to(device)
    MoNet = MomentNetwork().to(device)
    trainer = Trainer(G, MoNet, train_set, params_dict, device, learn_moments=True, scores=scores_dict)
    # trainer.generate_and_display(trainer.fixed_z, save=True, save_path=trainer.save_path + "generated_molm_cifar10_iter{}.png".format(1))
    trainer.train(save_images=True)

