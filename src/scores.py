from PIL import Image
import numpy as np
import torch
from torchvision import transforms, datasets

class Score():
    def __init__(self, device):
        self.device = device
        return None

    def __call__(self, batch):
        score = None
        return score

    
class InceptionScore(Score):
    transforms =  transforms.Compose([
                    transforms.Resize(299),
                    transforms.CenterCrop(299),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
    
    def __init__(self, model, device=None):
        super().__init__(device)
        self.model = model
        self.device = device
        if device:
            self.model = self.model.to(self.device)
        
    @classmethod
    def preprocess(cls, batch):
        return cls.transforms(batch)

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

def load_data(dataset):
    return datasets.CIFAR10(
    root=dataset,
    train=True,
    download=True, #download if non existant at the location 'root'
    transform=InceptionScore.transforms
)

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

