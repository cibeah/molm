import copy
from PIL import Image
import numpy as np
from scipy import linalg
import torch
from torchvision import transforms, datasets

#TODO: Add logger

class Score():
    def __init__(self, device):
        self.device = device
        return None

    def __call__(self, batch):
        score = None
        return score

class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class InceptionScore(Score):
    transforms_pipe =  transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    toPILImage = transforms.ToPILImage()
    
    def __init__(self, model, device=None):
        super().__init__(device)
        self.model = model
        self.device = device
        if device:
            self.model = self.model.to(self.device)
        
    @classmethod
    def preprocess(cls, batch):
        batch = (batch +1) / 2
        ##TODO: Implement transforms directly on Tensor without conversion to PIL Image
        imgs = [cls.transforms_pipe(cls.toPILImage(sample)).unsqueeze(0) for sample in batch]
        samples = torch.cat(
            imgs,
            dim=0
        )
        return samples

    def compute_score(self, probs, eps=1e-16):
        #compute marginal probabilities
        marginal_probs = probs.mean(0)

        #compute KL divergence for each image
        KL = probs * (np.log(probs + eps) - np.log(marginal_probs + eps))
        KL = KL.sum(1)

        #we average the KL divergences over the images
        KL_mean = KL.mean(0)
        score = np.exp(KL_mean)
        return score

    def __call__(self, batch):
        """
        returns the inception score for the batch of images
        """
        ##TODO: Check that it's a batch and throw error or unsqueeze
        ## batch = batch.unsqueeze(0)
        #batch = self.preprocess(batch)
        if self.device:
            batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch)
        probs = torch.nn.functional.softmax(output, dim=1).cpu()
        score = self.compute_score(probs)
        return float(score)


class FID(Score):

    def __init__(self, model, device=None):
        super().__init__(device)
        self.model = model
        self.device = device
        self.fitted = False
        self.reference_mu = np.zeros((1, 2048))
        self.reference_sigma = np.zeros((2048, 2048))
        if device:
            self.model = self.model.to(device)

    @staticmethod
    def make_fid_model(model):
        model_fid = copy.deepcopy(model)
        #replace last fc layer with identity layer
        model_fid.fc = Identity()
        model_fid.eval()
        return model_fid
    
    def fit(self, data, r=0.2):
        #TODO: maybe use r as the % of real data needed to compute real statistics
        """
        Compute real data features
        data: iterable data loader
        """

        for n, batch in enumerate(data):
            samples, _ = batch
            if self.device:
                samples = samples.to(self.device)
            with torch.no_grad():
                real_output = self.model(samples)
            if self.device:
                real_output = real_output.cpu().numpy()
            self.reference_mu += np.mean(real_output, 0)
            self.reference_sigma += np.cov(real_output, rowvar=False)
        #TODO: actual incremental calculation of sigma (mu is ok since all batches have the same size except 1)
        self.reference_mu /= len(data)
        self.reference_sigma /= len(data)
        self.fitted = True  

    def compute_score(self, ouput, eps=1e-16):
        """
        Compute FID on generated data.
        formula:  ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2))
        cf. https://github.com/bioinf-jku/TTUR/blob/master/fid.py
        Vectors must be on cpu.
        """
        #compute marginal probabilities
        if not self.fitted:
            raise TypeError("No reference statistics. Fit on real data before computing the score.")
        mu = np.mean(ouput, 0)
        sigma = np.cov(ouput, rowvar=False)
        covmean, _ = linalg.sqrtm(self.reference_sigma.dot(sigma), disp=False)

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        mu_diff = self.reference_mu - mu
        score = mu_diff.dot(mu_diff.T) + np.trace(self.reference_sigma + sigma - 2*covmean)
        return score

    def __call__(self, batch):
        if self.device:
            batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch)
        output = output.cpu().numpy()
        score = self.compute_score(output)
        return float(score)
    



#test
# model = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=True)
# model.eval()

# train_set = load_data("../data/CIFAR")
# loader = torch.utils.data.DataLoader(train_set,
#                                     shuffle=True,
#                                     batch_size=64)
# device = torch.device("cuda:0")
# IS = InceptionScore(model, "cuda:0")
# sample, _ = iter(loader).next()

# print("score:", IS(sample))

