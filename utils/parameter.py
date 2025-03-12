"""
Define a custom object whose attributes are channel scenario and training parameters. 

@author: Sueda Taner
"""
#%%
import numpy as np
import torch
from matplotlib import pyplot as plt

from matplotlib import cm
from matplotlib.colors import Normalize
#%% Create Parameter object

class Parameter:
    def __init__(self, ap_pos):
        # Initialize the Parameter object with the AP positions
        self.ap_pos = ap_pos
        # The UEs' information will be set by method set_UE_info
    
    # Training-related settings
    def set_training_params(self, W=16, in_features=256, hidden_features=(256,128,64), out_features=2, 
                   Tc=10, Tf=np.inf, segment_start_idcs=[0], M_t=1,
                   P_thresh=np.inf, M_p=0, M_b=1, lambda_b=0,
                   lr=1e-3, use_scheduler=0, scheduler_param=None,
                   num_epochs=50, batch_size=100):
        
        self.W = W # Number of time domain channel taps used for feature extraction   
        
        # Network dimension parameters
        self.in_features = in_features  # must match the number of features in dataset
        # For each entry, create a hidden layer with the corresponding number of units
        self.hidden_features = hidden_features
        self.out_features = out_features
          
        # Training parameters
        self.learning_rate = lr # training learning rate
        self.use_scheduler = use_scheduler # use learning rate scheduler
        self.scheduler_param = scheduler_param # lr schedueler parameter: reduce the lr at every this many epochs
        self.num_epochs = num_epochs # number of training epochs 
        self.batch_size = batch_size # training batch size

    # Simulation scenario-related settings    
    def set_UE_info(self, UE_pos, color_map=None, UE_time_stamps=None, dataset_idcs=None):
        self.UE_pos = UE_pos
        self.U = UE_pos.shape[0]
        if UE_time_stamps is None:
            self.UE_time_stamps = torch.arange(self.U)
        else:
            self.UE_time_stamps = UE_time_stamps
        if color_map is None:
            self.set_default_color_map()  # set the color map
        else:
            self.color_map = color_map
        self.dataset_idcs = dataset_idcs

    def set_default_color_map(self):
        # Coloring of the users
        color1 = (self.UE_pos[:, 0] - np.min(self.UE_pos[:, 0])).reshape((self.U, 1)) \
                 / (np.max(self.UE_pos[:, 0]) - np.min(self.UE_pos[:, 0]))
        color2 = (self.UE_pos[:, 1] - np.min(self.UE_pos[:, 1])).reshape((self.U, 1)) \
                 / (np.max(self.UE_pos[:, 1]) - np.min(self.UE_pos[:, 1]))
        color3 = np.zeros((self.U, 1))

        self.color_map = np.concatenate((color1, color2, color3), axis=-1)

    def plot_scenario(self, passive=False, dimensions='2d', title=None, colorbar=False):
        fig = plt.figure()
        if dimensions == '3d':
            ax = fig.add_subplot(projection='3d')

            ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], self.UE_pos[:, 2], marker='o', c=self.color_map)
            if passive:
                ax.scatter(self.TX_pos[:, 0], self.TX_pos[:, 1], self.TX_pos[:, 2], marker='+')

            ax.scatter(self.ap_pos[:, 0], self.ap_pos[:, 1], self.ap_pos[:, 2], marker='^')

            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
        elif dimensions == '2d':
            ax = fig.add_subplot()
            # ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], marker='o', c=self.color_map)
            if not colorbar:
                ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], s=0.5, c=self.color_map)
            else:
                ax.scatter(self.UE_pos[:, 0], self.UE_pos[:, 1], s=0.5, c=self.color_map, cmap='cool')
                normalize = Normalize(vmin=0, vmax=1) # vmin=min(self.color_map), vmax=max(self.color_map))
                cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalize, cmap=cm.cool), label='max. similarity value')

            if passive:
                ax.scatter(self.TX_pos[:, 0], self.TX_pos[:, 1], marker='+')
            
            ax.scatter(self.ap_pos[:, 0], self.ap_pos[:, 1], marker='^')
            for a in range(self.ap_pos.shape[0]):
                ax.annotate(a, (self.ap_pos[a, 0], self.ap_pos[a, 1]))
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_xticks([-15, -10, -5, 0, 5]) 
            ax.set_yticks([-16, -11, -6, -1]) 
            ax.set_ylim([-16, -1])
            ax.set_xlim([-15, 5])
            # ax.axis('equal')
            ax.set_aspect('equal', 'box')
            ax.grid()
        else:
            raise Exception('Undefined dimensions for plotting the scenario')
        if title is not None:
            plt.title(title)
      