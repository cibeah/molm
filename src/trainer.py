##TODO: dotenv instead of CAPS_PARAMS ! 
import os
import logging
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from torch.nn import MSELoss

from scores import InceptionScore

RUN_FOLDER = 'runs/run5'
SAVE_FOLDER_IMG = RUN_FOLDER + "/results/images/"
SAVE_FOLDER_CHECKPOINTS = RUN_FOLDER + "/checkpoints/"

logger = logging.Logger("trainer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class Trainer():
    
    def __init__(self, generator, moment_network, train_set, training_params, device=None, 
                scores=None, tensorboard=False):
        """
            generator: a nn.Module child class serving as a generator network
            moment_network: a nn.Module child class serving as the moment network
            loader: a training data loader
            scores: None, or a dict of shape {'name':obj} with score object with a __call__ function that returns a score
            
            training_params: dict of training parameters with:
                n0: number of objectives
                nm: number of moments trainig step
                ng: number of generating training steps
                lr: learning rate
                beta1 / beta2: Adam parameters 
                acw: activation wieghts
                alpha: the norm penalty parameter
                gen_batch_size: the batch size to train the generator
                mom_batch_size: the batch size to train the moment network
                eval_batch_size: the batch size to evaluate the generated
                eval_size: total number of generated samples on which to evaluate the scores

        """
        self.G = generator
        self.MoNet = moment_network
        self.train_set = train_set
        self.training_params = training_params
        self.nm = training_params["nm"]
        self.ng = training_params["ng"]
        self.no = training_params["no"]
        self.no_obj = 0 #current objective
        self.n_moments = training_params["n_moments"]
        self.gen_batch_size =  training_params["gen_batch_size"]
        self.eval_batch_size =  training_params["eval_batch_size"]
        self.learn_moments = training_params["learn_moments"]
        
        
        lr, beta1, beta2 = self.training_params["lr"], self.training_params["beta1"], self.training_params["beta2"]
        self.optimizerG = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizerM = optim.Adam(self.MoNet.parameters(), lr=lr, betas=(beta1, beta2))
 
        self.LM = []
        self.LG = []
        self.iter = 0
        self.device = device
        
        self.cross_entropy = F.binary_cross_entropy
        self.mse = MSELoss(reduction="sum")

        self.save_path_img = SAVE_FOLDER_IMG
        self.save_path_checkpoints = SAVE_FOLDER_CHECKPOINTS
        
        #to track the evolution of generated images from a single batch of noises
        self.fixed_z = torch.randn(20, self.G.dims[0], device=self.device)

        #monitoring the progress of the training with the evaluation scores
        self.scores = scores

        #monitoring through tensorboard
        if tensorboard:
            comment = ''.join(['{}={} '.format(key, training_params[key]) for key in training_params])
            self.tb = SummaryWriter(RUN_FOLDER, comment=comment)
            self.tb.add_graph(generator, self.fixed_z)

    def train_monet(self):
        #reshuffle training data
        loader = iter(torch.utils.data.DataLoader(self.train_set,
                                           shuffle=True,
                                          batch_size=self.training_params["mom_batch_size"]))
        for i in range(self.nm):
            batch = loader.next()
            samples, _ = batch
            samples = samples.to(self.device)
            samples = (samples * 2) - 1

            sample_size = samples.size(0)
            one_labels = torch.ones(sample_size, device=self.device)
            zero_labels = torch.zeros(sample_size, device=self.device)

            #generating latent vector 
            #self.dims = [Z_dim, H1_dim, H2_dim, H3_dim, X_dim]
            z = torch.randn(sample_size, self.G.dims[0], device=self.device)
            res = self.G(z)
            prob_trues, output_trues, _ = self.MoNet(samples)
            prob_gen, output_gen, _ = self.MoNet(res)

            prob_trues, prob_gen = prob_trues.squeeze(), prob_gen.squeeze()
            LM_samples = self.cross_entropy(prob_trues, one_labels)
            LM_gen = self.cross_entropy(prob_gen, zero_labels)
            LM = LM_samples + LM_gen

            #We now need to compute the gradients to add the regularization term
            mean_output = output_trues.mean()
            self.optimizerM.zero_grad()
            grad_monet = self.MoNet.get_gradients(mean_output)
            #This is the sum of gradients, so we divide by the batch size
            grad_monet = (grad_monet / sample_size).squeeze()
            grad_norm =  torch.dot(grad_monet, grad_monet)
            LM = LM_samples + LM_gen + self.training_params["alpha"] * ((grad_norm - 1)**2)
            #LM = LM_samples + LM_gen
            #print("LM loss: {:.4}".format(float(LM)))
            #Add to tensorboard
            if self.tb:
                    self.tb.add_scalar('LossMonet/objective_{}'.format(self.no_obj+1), float(LM), i+1)
            self.LM.append(float(LM))
            if i%50 == 0:
                logger.info("Moment Network Iteration {}/{}: LM: {:.6}".format(i+1, self.nm, LM.item()))
            
            self.optimizerM.zero_grad()
            LM.backward()
            self.optimizerM.step()

            del grad_monet
            del batch
  
    def eval_true_moments(self):
        loader = torch.utils.data.DataLoader(self.train_set,
                                           shuffle=True,
                                          batch_size=self.training_params["mom_batch_size"])
        #Calculate the moment vector over the entire dataset:
        moments = torch.zeros(self.n_moments, device=self.device)
        for i, batch in enumerate(loader):
            #if i % 100 == 0:
             #   print("Computing real data Moment Features... {}/{}".format(i+1, len(loader)))
            samples, _ = batch
            samples = samples.to(self.device)
            sample_size = samples.size(0)
            #Scaling true images to tanh activation interval:
            samples = (samples * 2) - 1
            self.optimizerM.zero_grad()
            moments_b = self.MoNet.get_moment_vector(samples, sample_size, weights = self.training_params["activation_weight"], detach=True)   
            moments = ((i) * moments + moments_b) / (i+1)
            del batch
            del samples
            del moments_b
        return moments
                                        
                                        
    def train_generator(self, true_moments): 
        for i in range(self.ng):
            #moments_gz = torch.zeros(n_moments, device=self.device)  
            #print(i)
            #if i%225 ==0 :
             #   print("Computing Monte Carlo estimate of generated data Moment Features... {}/{}".format(i+1, 5))
           
            
            z = torch.randn(self.gen_batch_size, self.G.dims[0], device=self.device)
            res = self.G(z)
            self.optimizerM.zero_grad()
            moments_gz = self.MoNet.get_moment_vector(res, self.gen_batch_size, weights = self.training_params["activation_weight"])
            #moments_gz = ((i) * moments_gz + moments_z) / (i+1)
        
            del z
            del res

            #LG = torch.dot(true_moments - moments_gz, true_moments - moments_gz) #equivalent to dot product of difference
            LG = self.mse(true_moments, moments_gz)
            #Add to tensorboard
            if self.tb:
                    self.tb.add_scalar('LossGenerator/objective_{}'.format(self.no_obj+1), float(LG), i+1)
            self.LG.append(float(LG))
            if i%100 == 0:
                logger.info("Generator Iteration {}/{}: LG: {:.6}".format(i+1, self.ng, LG.item()))
            self.optimizerG.zero_grad()
            LG.backward() 
            self.optimizerG.step()
            
            del moments_gz
            
    def generate_and_display(self, z, save=False, save_path=None):
        #Visualizing the generated images
        examples = self.G(z).detach().cpu()
        examples = examples.reshape(-1, 3, 32, 32)
        examples = (examples +1) / 2
        grid = torchvision.utils.make_grid(examples, nrow=10) # 10 images per row
        #Add to tensorboard
        if self.tb:
            self.tb.add_image('generated images', grid, self.no_obj)
        fig = plt.figure(figsize=(15,15))
        plt.imshow(np.transpose(grid, (1,2,0)))
        if save:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close(fig)


    def eval(self):
        logger.info("Evaluating generated images with scores: {}".format(self.scores.keys()))
        scores_dict = self.scores
        n_loops = self.training_params["eval_size"] // self.eval_batch_size
        results = dict(zip(scores_dict.keys(), [0]*len(scores_dict)))
        for i in range(n_loops):
            with torch.no_grad():
                z = torch.randn(self.eval_batch_size, self.G.dims[0], device=self.device)
                res = self.G(z).cpu()
            samples = InceptionScore.preprocess(res)
            for score in scores_dict:
                scoring = scores_dict[score]
                results[score] += scoring(samples)
        for score in scores_dict:
            results[score] /= n_loops
        return results
        

    def train(self, save_images=False):
        if not self.learn_moments:
            true_moments = self.eval_true_moments()
        
        for i in range(self.no):
            #Track the no of objectives solved
            self.no_obj = i
            
            start = time.time()
            if self.learn_moments:
                logger.info("Training Moment Network...")
                self.train_monet()
                logger.info("Evaluating true moments value...")
                true_moments = self.eval_true_moments()
            logger.info("Training Generator")
            self.train_generator(true_moments)
            self.iter += 1
            stop = time.time()
            duration = (stop - start)/60
            
            if self.learn_moments:
                logger.info("Objective {}/{} - {:.2} minutes: LossMonet: {:.6} LossG: {:.6}".format(i+1, self.no, duration, self.LM[-1], self.LG[-1]))
            else:
                logger.info("Objective {}/{} - {:.2} minutes: LossG: {:.6}".format(i+1, self.no, duration, self.LG[-1]))

            self.generate_and_display(self.fixed_z, save=save_images, 
                                      save_path=self.save_path_img + "generated_molm_cifar10_iter{}.png".format(i))
            
            if i%5 == 0:
                logger.info("Saving model ...")
                save_path_checkpoints = self.save_path_checkpoints + "molm_cifar10_iter{}.pt".format(i)
                save_dict = {
                            'monet_state_dict': self.MoNet.state_dict(),
                            'generator_state_dict': self.G.state_dict(),
                            'optimizerG_state_dict': self.optimizerG.state_dict(),
                            'objective': i+1,
                            'last_lossG': self.LG[-1]
                            }
                if self.learn_moments:
                    save_dict["last_lossM"] = self.LM[-1]
                    save_dict["optimizerM_state_dict"] = self.optimizerM.state_dict()
                
                torch.save(save_dict, save_path_checkpoints)

                if self.scores:
                    scores = self.eval()
                    logger.info(scores)
                    #Add to tensorboard
                    if self.tb:
                        for score in scores:
                            self.tb.add_scalar('Scores/{}'.format(score), scores[score], i+1)

            # Updating data on tensorboard
            if self.tb:
                for name, param in self.G.named_parameters():
                    self.tb.add_histogram('generator.{}'.format(name), param, i+1)
                    self.tb.add_histogram('generator.{}.grad'.format(name), param.grad, i+1)
                for name, param in self.MoNet.named_parameters():
                    self.tb.add_histogram('momentNetwork.{}'.format(name), param, i+1)
                    self.tb.add_histogram('momentNetwork.{}.grad'.format(name), param.grad, i+1)
            
def get_n_params(Network):
    n_params = 0
    sizes = {}
    for param_tensor in Network.state_dict():
        sizes[param_tensor] = Network.state_dict()[param_tensor].size()
        n_params += Network.state_dict()[param_tensor].numel()
    return n_params, sizes