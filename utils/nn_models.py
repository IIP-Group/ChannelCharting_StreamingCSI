"""
Neural network models to be used in channel charting.
First, define a generic fully connected network. Then, define "model" classes to train and evaluate the network.

@author: Sueda Taner
"""
#%%
import torch
import torch.optim
from torch import nn
from tqdm import trange
import typing

#%% The class for a generic fully connected network to be used in all methods

class FCNet(nn.Module):
    """
    A class for a generic fully connected network.
    This network is the building block for all channel charting methods.
    
    """
    
    def __init__(self, in_features: int, hidden_features: typing.Tuple[int, ...], out_features: int, add_BN=False):
        """
        Create a fully connected network.

        Parameters
        ----------
        in_features : int
            Input features' dimension.
        hidden_features : typing.Tuple[int, ...]
            Tuple where each entry corresponds to a hidden layer with the 
            corresponding feature dimension.
        out_features : int
            Output features' dimension.
        """
        super().__init__()

        feature_sizes = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(feature_sizes) - 1

        self.layers = nn.ModuleList()
        for idx in range(num_affine_maps):
            self.layers.append(nn.Linear(feature_sizes[idx], feature_sizes[idx + 1]))
            torch.nn.init.xavier_uniform_(self.layers[-1].weight) # Glorot initialization
            if add_BN:
                self.layers.append(nn.BatchNorm1d(feature_sizes[idx + 1]))
        self.num_layers = len(self.layers)

        self.activation = nn.ReLU()
        self.add_BN = add_BN

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fully connected network.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        current_features : torch.Tensor
            Output of the network.

        """
        current_features = x
        
        for idx, current_layer in enumerate(self.layers[:-1]):
            current_features = current_layer(current_features)
            if not self.add_BN: current_features = self.activation(current_features)
            else:
                if idx % 2 == 1: # activation after BN
                    current_features = self.activation(current_features)
        # Last layer has no BN or activation
        current_features = self.layers[-1](current_features) # last layer has no activation
        
        return current_features      

#%% Functions that are shared among multiple classes of models

def evaluate(model, X: torch.Tensor) -> torch.Tensor:
    """
    Set a model's network in evaluation mode and make a forward pass.

    Parameters
    ----------
    X : torch.Tensor
        Dataset (i.e, the CSI features).

    Returns
    -------
    X : torch.Tensor
        Channel chart.

    """        
    model.network.eval()
    
    test_loader = torch.utils.data.DataLoader(
        X, batch_size=model.par.batch_size, shuffle=False) 
    
    Y = torch.zeros((0, model.par.out_features), device=model.device)
    for batch_x in test_loader:
        batch_x = batch_x.to(model.device)
        Y = torch.concatenate((Y, model.network(batch_x)))
    return Y

    
#%% Models to train the FCNet using different learning approaches

#%%
class SupervisedModel:
    """
    A class to train an FCNet using ground truth labels in an MSE loss. 
    
    Attributes
    ----------
    device : torch.device
        Device to use for training or evaluation, e.g. torch.device("cpu") or torch.device("cuda:0").
    par : Parameter
        Simulation scenario and training-related parameters are all stored in this object.
    
    """
    
    def __init__(self, device, par):
        
        self.device = device
        self.par = par
        
        self.network = FCNet(self.par.in_features, self.par.hidden_features, self.par.out_features)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.par.learning_rate)
        self.network.to(device)
        
    def print_params(self):
        if self.par.use_scheduler == 0:
            print('Tc:', self.par.Tc, ', M:', self.par.M_t,  
              'use_sch:', self.par.use_scheduler)   
        else:
            print('Tc:', self.par.Tc, ', M:', self.par.M_t,  
              'use_sch:', self.par.use_scheduler, ', sche param:', self.par.scheduler_param)
    
    def train(self, dataset: torch.utils.data.Dataset):
        """
        Train the network based on model parameters.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Training dataset consisting of CSI features and ground-truth positions.
            
        Returns
        -------
        loss_per_epoch : np.array
            Average loss per batch at each epoch for training visualization.

        """
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.par.batch_size, shuffle=True) 
        self.network.train()  # set the module in training mode

        num_batches = len(train_loader)

        progress_bar = trange(self.par.num_epochs)  # just trying if this makes shit slow
        loss_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
            
        for epoch_idx in progress_bar:
            sum_loss_per_batch = torch.tensor(0.0, device=self.device)
            
            for batch_idx, (batch_x,batch_y) in enumerate(train_loader):
                # Set all the gradients to zero
                self.network.zero_grad()
                # Make a forward pass
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_y_hat = self.network(batch_x)
                
                loss = torch.mean(torch.linalg.norm(batch_y_hat - batch_y,2,-1)**2)                 
                
                # Backpropagate to get the gradients
                loss.backward()
                self.optimizer.step()
                sum_loss_per_batch = sum_loss_per_batch + loss.detach()  
                
            loss_per_epoch[epoch_idx] = sum_loss_per_batch / num_batches
        loss_per_epoch = loss_per_epoch.cpu().numpy()

        return loss_per_epoch

#%%
class SiameseModel:
    """
    A class to train an FCNet as a Siamese network.
    
    Attributes
    ----------
    device : torch.device
        Device to use for training or evaluation, e.g. torch.device("cpu") or torch.device("cuda:0").
    par : Parameter
        Simulation scenario and training-related parameters are all stored in this object.
    
    """
    
    def __init__(self, device, par):
        
        self.device = device
        self.par = par
        
        self.network = FCNet(self.par.in_features, self.par.hidden_features, self.par.out_features, add_BN=True)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.par.learning_rate)
        self.network.to(device)
        if par.use_scheduler == 1:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, par.scheduler_param) 

    def print_params(self):
        if self.par.use_scheduler == 0:
            print('use_sch:', self.par.use_scheduler)   
        else:
            print('use_sch:', self.par.use_scheduler, ', sche param:', self.par.scheduler_param)
    
    def train(self, X, D_cs, loss_version=1):
        """
        Train the network based on model parameters.
        
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Training dataset consisting of CSI features and geodesic distances
            
        Returns
        -------
        loss_per_epoch : np.array
            Average loss per batch at each epoch for training visualization.

        """
        train_loader = torch.utils.data.DataLoader(
            torch.arange(X.shape[0], dtype=int), batch_size=self.par.batch_size, shuffle=True) 
        self.network.train()  # set the module in training mode

        num_batches = len(train_loader)

        progress_bar = trange(self.par.num_epochs)  # just trying if this makes shit slow
        loss_per_epoch = torch.zeros(self.par.num_epochs, device=self.device)
        
        if loss_version == 2:
            dissim_margin = torch.quantile(D_cs.flatten(), 0.01)
            
        for epoch_idx in progress_bar:
            sum_loss_per_batch = torch.tensor(0.0, device=self.device)
            
            for batch_idx, (dataset_idcs) in enumerate(train_loader):
                # Set all the gradients to zero
                self.network.zero_grad()
                # Transfer to device
                batch_x = X[dataset_idcs].to(self.device)
                batch_D = D_cs[dataset_idcs][:,dataset_idcs].to(self.device)
                # Make a forward pass
                batch_y = self.network(batch_x)
                # Calculate the loss
                if loss_version == 1:
                    loss = torch.mean((batch_D - torch.linalg.norm(batch_y[:,None]-batch_y, dim=2))**2)
                elif loss_version == 2:
                    loss = torch.mean((1/(batch_D+dissim_margin)) * (batch_D - torch.linalg.norm(batch_y[:,None]-batch_y, dim=2))**2)
                
                # Backpropagate to get the gradients
                loss.backward()
                self.optimizer.step()
                sum_loss_per_batch = sum_loss_per_batch + loss.detach()  # gpu # Should this be detach or item() or.?
                    
            loss_per_epoch[epoch_idx] = sum_loss_per_batch / num_batches
            if self.par.use_scheduler == 1: self.scheduler.step()
            elif self.par.use_scheduler == 2: self.scheduler.step(loss_per_epoch[epoch_idx])
            
        loss_per_epoch = loss_per_epoch.cpu().numpy()

        return loss_per_epoch
    
