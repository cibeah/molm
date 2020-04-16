##TODO: ADD Timing ! 
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.nn import MSELoss


SAVE_FOLDER = "results/images/"

class Trainer():
    
    def __init__(self, generator, moment_network, train_set, training_params, device=None):
        """
            generator: a nn.Module child class serving as a generator network
            moment_network: a nn.Module child class serving as the moment network
            loader: a training data loader
            
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

        """
        self.G = generator
        self.MoNet = moment_network
        self.train_set = train_set
        self.training_params = training_params
        self.nm = training_params["nm"]
        self.ng = training_params["ng"]
        self.no = training_params["no"]
        self.n_moments = training_params["n_moments"]
        self.gen_batch_size =  training_params["gen_batch_size"]
        
        
        lr, beta1, beta2 = self.training_params["lr"], self.training_params["beta1"], self.training_params["beta2"]
        self.optimizerG = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        self.optimizerM = optim.Adam(self.MoNet.parameters(), lr=lr, betas=(beta1, beta2))
 
        self.memory_allocated = []
        self.LM = []
        self.LG = []
        self.iter = 0
        self.device = device
        
        self.cross_entropy = F.binary_cross_entropy
        self.mse = MSELoss(reduction="sum")

        self.save_path = SAVE_FOLDER
        
        #to track the evolution of generated images from a single batch of noises
        self.fixed_z = torch.randn(20, self.G.dims[0], device=self.device)
        
    def train_monet(self):
        #reshuffle training data
        loader = iter(torch.utils.data.DataLoader(self.train_set,
                                           shuffle=True,
                                          batch_size=self.training_params["mom_batch_size"]))
     
        torch.cuda.empty_cache()
        memory_allocated = []
        
        for i in range(self.nm):
            #Monitor GPU Mmemory
            memory_allocated.append(torch.cuda.memory_allocated())
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

            self.LM.append(float(LM))
            if i%50 == 0:
                print("Moment Network Iteration {}/{}: LM: {:.6}".format(i+1, self.nm, LM.item()))
            
            self.optimizerM.zero_grad()
            LM.backward()
            self.optimizerM.step()
            memory_allocated.append(torch.cuda.memory_allocated())

            del grad_monet
            del batch
            torch.cuda.empty_cache()
        self.memory_allocated.append(memory_allocated)
  
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
            torch.cuda.empty_cache()
        return moments
                                        
                                        
    def train_generator(self, true_moments): 
        torch.cuda.empty_cache()
        memory_allocated = []
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
        
            memory_allocated.append(torch.cuda.memory_allocated())
            del z
            del res
            
            torch.cuda.empty_cache()
            memory_allocated.append(torch.cuda.memory_allocated())

            LG = torch.dot(true_moments - moments_gz, true_moments - moments_gz) #equivalent to dot product of difference 
            self.LG.append(float(LG))
            if i%100 == 0:
                print("Generator Iteration {}/{}: LG: {:.6}".format(i+1, self.ng, LG.item()))
            self.optimizerG.zero_grad()
            LG.backward() 
            self.optimizerG.step()
            
            del moments_gz
            torch.cuda.empty_cache()
        self.memory_allocated.append(memory_allocated)
            
    def generate_and_display(self, z, save=False, save_path=None):
        #Visualizing the generated images
        examples = self.G(z).detach().cpu()
        examples = examples.reshape(-1, 3, 32, 32)
        examples = (examples +1) / 2
        grid = torchvision.utils.make_grid(examples, nrow=10) # 10 images per row
        plt.figure(figsize=(15,15))
        plt.imshow(np.transpose(grid, (1,2,0)))
        if save:
            plt.savefig(save_path)
        else:
            plt.show()

    def train(self, save_images=False):
        for i in range(self.no):
            print("Training Moment Network...")
            self.train_monet()
            print("Evaluating true moments value...")
            true_moments = self.eval_true_moments()
            print("Training Generator")
            self.train_generator(true_moments)
            self.iter += 1
            
            
            print("Objective {}/{} : LossMonet: {:.6} LossG: {:.6}".format(i+1, self.no, self.LM[-1], self.LG[-1]))
            self.generate_and_display(self.fixed_z, save=save_images, 
                                      save_path=self.save_path + "generated_molm_cifar10_iter{}.png".format(i))
            
            if i%10 == 1:
                print("Saving model ...")
                save_path = "checkpoints/molm_cifar10_iter{}.pt".format(i)
                torch.save({
                            'monet_state_dict': self.MoNet.state_dict(),
                            'generator_state_dict': self.G.state_dict(),
                            'optimizerM_state_dict': self.optimizerM.state_dict(),
                            'optimizerG_state_dict': self.optimizerG.state_dict(),
                            'objective': i+1,
                            'last_lossM': self.LM[-1],
                            'last_lossG': self.LG[-1]

                            }, save_path)
            