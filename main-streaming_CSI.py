"""
--Channel Charting for Streaming CSI Data--

Maintain a small core memory of CSI using a min-max similarity criterion.
and train a neural network for channel charting based on this memory.
Test the channel charting network on a separate dataset.
"""
#%
import os
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from copy import deepcopy
from datetime import datetime
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import progressbar
import multiprocessing as mp

from utils.parameter import Parameter
from utils.nn_models import SupervisedModel,  SiameseModel, evaluate
from utils.cc_helpers import plot_chart, evaluate_cc

#%
# Set the device to train on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Specify your runID
runID = 'your_runID'

# Set neural network type: Use 'Siamese' for channel charting and 'Supervised' for supervised positioning
network_type = 'Siamese' # 'Siamese' or 'Supervised' 

# Use geodesic distances for training the Siamese network
geodesic = 1 # 0 or 1

# Set the number of taps (<=13) of the time domain CSI to use for feature extraction
W = 13

# Dataset reduction-related settings:
# Set training segment = segment of streaming CSI. Use segment 1 (cf02 dataset)
tr_seg = 1 
# Chop the last 1k of data so that the trajectory does not go back: 17516, otherwise full segment 1: 18516 
cut_idx = 17516 
# Set the curation strategy: 1 for random with prob p_replace, 2 for minimizing max cos sim in the data
replace_criterion = 2 
# Buffer length: how many samples should we store in the memory
buf_len = 1000 #
# For the random curation strategy, the probability that we replace a sample in memory
p_replace = 0.5

# Channel charting test is on segment 2 (cf03 dataset)
test_seg = 2

# Network size parameters 
hidden_features = (256, 128, 64, 32, 16) 

# Training parameters
use_scheduler, scheduler_param = 1, 200
num_epochs, batch_size = 2000, 128
lr = 1e-2 

# Specify saving, plotting & evaluation preferences
save = 1 # 0 or 1 # Save training parameters and trained model weights
plot_loss_per_epoch = True
eval_on_training_set = True # Evaluate the trained model on the training set
eval_on_test_set = True # Evaluate the trained model on the test set


#% Fix the seed - you might get slightly different results from the paper
torch.manual_seed(0)
np.random.seed(0)
np.set_printoptions(precision=3)

#% Import the AP antenna positions and initialize the Parameter object
data_path = './data'
# Use AP positions for ground truth visualization
ap_pos = np.array([[  2.675, -13.897,   1.56 ],
                   [-11.25 ,  -9.689,   1.534],
                   [ -1.531, -15.059,   1.577],
                   [-12.684,  -4.483,   1.79 ]])
par_all = Parameter(ap_pos)

csi_time_domain = np.load(f'{data_path}/csi_calibrated_13taps_per_ap.npy')[:,:,:W]
H = np.transpose(csi_time_domain, (0, 3, 1, 2))
H = np.reshape(H, (H.shape[0], H.shape[1], -1))
H = np.transpose(H, (0, 2, 1))

UE_pos = np.load(f'{data_path}/pos_tachy.npy')
timestamps = np.load(f'{data_path}/time.npy')
n_seg = np.load(f'{data_path}/n_segment.npy').astype(np.int32) # [18516, 23478, 41991
segment_start_idcs = np.cumsum(np.insert(n_seg, 0, 0))[:-1]

#% Core memory curation
A = par_all.ap_pos.shape[0] # the number of APs
Mr = int(H.shape[1] / A) # the number of antennas per AP

# par_all stores the ground truth position of the whole dataset
par_all.set_UE_info(UE_pos, None, timestamps) # to get the most detailed color map

# Compute all features at once
Htorch = torch.from_numpy(H.reshape(H.shape[0], -1))
X_all = torch.abs(Htorch)/torch.linalg.norm(Htorch, dim=-1, keepdim=True) # Extract features from the CSI

#% Incoming CSI's dataset idcs
if tr_seg == 1: 
    incoming = np.arange(0, cut_idx, dtype=int)
else: raise RuntimeError('Undefined training segment')

par_inc = deepcopy(par_all)
par_inc.set_UE_info(UE_pos[incoming], par_all.color_map[incoming]) # default settings to get the color map for all   
par_inc.plot_scenario(passive=False, dimensions='2d', title='Ground truth positions of incoming CSI')

H_inc, UE_pos_inc = torch.from_numpy(H[incoming]), UE_pos[incoming]
H_inc = H_inc.reshape(H_inc.shape[0], -1)

#% 
def cosine_sim(X, Y=None, zero_diag=False):
    norm_X = torch.linalg.norm(X, dim=-1, keepdim=True) 
    if Y is None: # Compute cos sim in X
        cs = torch.abs(torch.conj(X) @ X.T)  / (norm_X * norm_X.T)
        Y = X
    else:
        cs = (torch.abs(torch.conj(X) @ Y.T)   
                     / (norm_X * torch.linalg.norm(Y, dim=-1, keepdim=True).T))
    if zero_diag:
        diag_idcs = np.arange(min(X.shape[0], Y.shape[0]), dtype=int)
        cs[diag_idcs, diag_idcs] = torch.zeros((len(diag_idcs)))
    return cs
    
par_tr = deepcopy(par_inc)

# Subset selection from incoming features
X_inc = X_all[incoming]
        
kk_title = 'Subset picking - min max cos sim algo update idx:' if replace_criterion == 2 else 'Subset picking - rnd replacement algo update idx:'

# The memory first buffers all of the first buf_len samples
bufd_UEs = np.arange(0, buf_len, dtype=int) # buffered UEs (indices from the incoming set)
bufd_UE_pos = np.zeros((buf_len,2)) # buffered UEs' gt positions (for visualization)
bufd_X0 = X_inc[:buf_len] # Extract features from the first CSI
X_onl = X_inc[buf_len:] # Extract features from the later CSI

bufd_X = bufd_X0 + 0 # the updated centroids
norm_X = torch.linalg.norm(bufd_X, dim=-1, keepdim=True) 
cos_sims = cosine_sim(bufd_X, zero_diag=True)
vals, cols = torch.max(cos_sims, dim=0)
max_cos_sim, idx_del = torch.max(vals, dim=0)
for kk, x in enumerate(X_onl):
    if replace_criterion == 1:
        replace = np.random.binomial(size=1, n=1, p=p_replace)
        if replace:
            # Replace a rnd sample
            idx_del = np.random.choice(np.arange(buf_len))
            bufd_UEs = np.delete(bufd_UEs, idx_del)
            bufd_UEs = np.append(bufd_UEs, kk+buf_len)
        if kk == X_onl.shape[0]-1: # get the centroids in the end
            bufd_X = X_inc[bufd_UEs] # Extract features from the CSI
    
    elif replace_criterion == 2:   
        new_cos_sims = (torch.abs(torch.conj(bufd_X) @ x[:,None])   
                          / (norm_X * torch.linalg.norm(x)))
        cos_sims = cosine_sim(bufd_X, x[None,:])
        cur_cos_sim = torch.max(cos_sims)
        if cur_cos_sim < max_cos_sim:
            bufd_X = torch.cat((bufd_X[:idx_del], bufd_X[idx_del+1:]), dim=0)
            bufd_X = torch.cat((bufd_X, x[None,:]), dim=0) 
            bufd_UEs = np.delete(bufd_UEs, idx_del)
            bufd_UEs = np.append(bufd_UEs, kk+buf_len)
            cos_sims = cosine_sim(bufd_X, zero_diag=True)
            max_cos_sim = torch.max(cos_sims)
            row, col = divmod(torch.argmax(cos_sims).item(), cos_sims.shape[1])
            idx_del = row if np.random.randint(0, 2) else col
    else: raise RuntimeError('Undefined replace criterion')
    
    if (kk + 1000) % 6000 == 0 or kk == X_onl.shape[0]-1:
        print('Online update idx:', kk)
        heat_vals = torch.max(torch.nn.functional.relu(cos_sims), dim=1, keepdim=True).values # are betw 0, 1
        # Coloring of the users for the cos sim
        par_tr.set_UE_info(UE_pos_inc[bufd_UEs], heat_vals.numpy())
        par_tr.plot_scenario(passive=False, dimensions='2d', title='Cos sim heat map at update idx:' + str(kk), colorbar=True)
            
X = bufd_X # to keep the notation consistent 

# Core memory curation is finished!

#% Now it's time for channel charting/positioning with the buffered features in core memory

# Separate train and test set data
tr_UEs = incoming[bufd_UEs]
# Get the test sample indices
te_UEs = segment_start_idcs[1] + np.arange(n_seg[1], dtype=int)

H_tr, UE_pos_tr, timestamps_tr = H[tr_UEs], UE_pos[tr_UEs], timestamps[tr_UEs]
H_te, UE_pos_te = H[te_UEs], UE_pos[te_UEs]

#% Parameter object for the training set
# Overwrite par_tr to get the coloring
par_tr.set_UE_info(UE_pos_tr, par_all.color_map[tr_UEs], par_all.UE_time_stamps[tr_UEs], tr_UEs)
par_tr.plot_scenario(passive=False, dimensions='2d', title='Ground truth positions of the training set')
print('Num of training samples: ', H_tr.shape[0])

csi_time_domain_all, timestamps_all = csi_time_domain, timestamps # renaming to match the tutorial

#% Compute the dissimilarity metric to be used in the Siamese network
csi_time_domain, timestamps = csi_time_domain_all[tr_UEs], timestamps_all[tr_UEs]
adp_dissimilarity_matrix = np.zeros((csi_time_domain.shape[0], csi_time_domain.shape[0]), dtype=np.float32)

def adp_dissimilarities_worker(todo_queue, output_queue):
	def adp_dissimilarities(index):
		# h has shape (arrays, antennas, taps), w has shape (datapoints, arrays, antennas, taps)
		h = csi_time_domain[index,:,:,:]
		w = csi_time_domain[index:,:,:,:]
			
		dotproducts = np.abs(np.einsum("bmt,lbmt->lbt", np.conj(h), w, optimize = "optimal"))**2
		norms = np.real(np.einsum("bmt,bmt->bt", h, np.conj(h), optimize = "optimal") * np.einsum("lbmt,lbmt->lbt", w, np.conj(w), optimize = "optimal"))
		
		return np.sum(1 - dotproducts / norms, axis = (1, 2))

	while True:
		index = todo_queue.get()

		if index == -1:
			output_queue.put((-1, None))
			break
		
		output_queue.put((index, adp_dissimilarities(index)))

count = 0
with progressbar.ProgressBar(max_value = csi_time_domain.shape[0]**2) as bar:
	todo_queue = mp.Queue()
	output_queue = mp.Queue()

	for i in range(csi_time_domain.shape[0]):
		todo_queue.put(i)
	
	for i in range(mp.cpu_count()):
		todo_queue.put(-1)
		p = mp.Process(target = adp_dissimilarities_worker, args = (todo_queue, output_queue))
		p.start()

	finished_processes = 0
	while finished_processes != mp.cpu_count():
		i, d = output_queue.get()

		if i == -1:
			finished_processes = finished_processes + 1
		else:
			adp_dissimilarity_matrix[i,i:] = d
			adp_dissimilarity_matrix[i:,i] = d
			count = count + 2 * len(d) - 1
			bar.update(count)

dissimilarity_matrix = adp_dissimilarity_matrix

#% Geodesic distances
if geodesic:
    n_neighbors = 20
    
    nbrs_alg = NearestNeighbors(n_neighbors = n_neighbors, metric="precomputed", n_jobs = -1)
    nbrs = nbrs_alg.fit(dissimilarity_matrix)
    nbg = kneighbors_graph(nbrs, n_neighbors, metric = "precomputed", mode="distance")
    
    dissimilarity_matrix_geodesic = np.zeros((nbg.shape[0], nbg.shape[1]), dtype = np.float32)
    
    def shortest_path_worker(todo_queue, output_queue):
    	while True:
    		index = todo_queue.get()
    
    		if index == -1:
    			output_queue.put((-1, None))
    			break
    
    		d = dijkstra(nbg, directed=False, indices=index)
    		output_queue.put((index, d))
    
    count = 0
    with progressbar.ProgressBar(max_value = nbg.shape[0]**2) as bar:
    	todo_queue = mp.Queue()
    	output_queue = mp.Queue()
    
    	for i in range(nbg.shape[0]):
    		todo_queue.put(i)
    	
    	for i in range(mp.cpu_count()):
    		todo_queue.put(-1)
    		p = mp.Process(target = shortest_path_worker, args = (todo_queue, output_queue))
    		p.start()
    
    	finished_processes = 0
    	while finished_processes != mp.cpu_count():
    		i, d = output_queue.get()
    
    		if i == -1:
    			finished_processes = finished_processes + 1
    		else:
    			dissimilarity_matrix_geodesic[i,:] = d
    			count = count + len(d)
    			bar.update(count)
    dissimilarity_matrix = dissimilarity_matrix_geodesic
    dissimilarity_margin = np.quantile(dissimilarity_matrix_geodesic, 0.01) # @sue not used yet

D = torch.from_numpy(dissimilarity_matrix)  

#% Train the neural network  
X_tr = X_all[tr_UEs]

# Store the network parameters in the most extensive way, some of them may go unused depending on learning_type
par_tr.set_training_params(W=W, in_features=X_tr.shape[1], hidden_features=hidden_features, out_features=2, 
                    lr=lr, use_scheduler=use_scheduler, scheduler_param=scheduler_param,
                    num_epochs=num_epochs, batch_size=batch_size) # enough for Siamese

if network_type == 'Siamese':
    model = SiameseModel(device, par_tr)
    loss_per_epoch = model.train(X_tr, D)
elif network_type == 'Supervised':
    train_set = torch.utils.data.TensorDataset(X_tr, torch.from_numpy(par_tr.UE_pos))  
    model = SupervisedModel(device, par_tr)
    loss_per_epoch = model.train(train_set)
    
print('\nTraining completed!')

if plot_loss_per_epoch:
    plt.figure()
    plt.plot(np.log10(loss_per_epoch))
    plt.xlabel('epoch')
    plt.title('Avg log loss of batches in each epoch')
    
#% Evaluate the model on the training set
if eval_on_training_set:
    Y = evaluate(model, X_tr)
    Y = Y.cpu().detach().numpy()
       
    # Calculate eval metrics
    ks,rd = evaluate_cc(par_tr.UE_pos, Y, 'KS-RD')
    ks_str, rd_str = np.around(ks, decimals=3), np.around(rd, decimals=3)
    print('---\nTraining set performance:\nKS:', ks_str, 'RD:', rd_str)
    fig = plot_chart(Y, par_tr.color_map, False, None, False, title=f'Channel chart of the training set. KS={ks_str} RD={rd_str}')
    
#% Test the trained model 

# Take every other sample in the test set for a quicker run
skip = 2

par_te = None
if eval_on_test_set: 
    par_te = deepcopy(par_tr)
    
    par_te.set_UE_info(UE_pos[te_UEs[::skip]], par_all.color_map[te_UEs[::skip]])
    par_te.plot_scenario(passive=False, dimensions='2d', title='Ground truth positions of the test set')
       
    X = X_all[te_UEs[::skip]]
    Y = evaluate(model, X)
    Y = Y.cpu().detach().numpy()
       
    # Calculate eval metrics
    ks,rd = evaluate_cc(par_te.UE_pos, Y, 'KS-RD')
    ks_str, rd_str = np.around(ks, decimals=3), np.around(rd, decimals=3)
    print('---\nTest set performance:\nKS:', ks_str, 'RD:', rd_str)
    tw, ct = evaluate_cc(par_te.UE_pos, Y, 'TW-CT')
    tw_str, ct_str = np.around(tw, decimals=3), np.around(ct, decimals=3)
    fig = plot_chart(Y, par_te.color_map, False, None, False, title=f'Channel chart of the test set. KS={ks_str} RD={rd_str} TW={tw_str} CT={ct_str}')
    
    
#% Save the used parameters and results 
save_path = './results'
os.makedirs(save_path, exist_ok=True)

now = datetime.now() # datetime object containing current date and time
date_str = now.strftime("%m_%d-%H_%M")
saveID = runID + '--'+ date_str
if save:
    os.makedirs(f'{save_path}/network_params', exist_ok=True)
    torch.save(model.network.state_dict(), f'{save_path}/network_params/{saveID}.pth')
    
    os.makedirs(f'{save_path}/training_params', exist_ok=True)
    f = open(f'{save_path}/training_params/{saveID}.pckl', 'wb')
    pickle.dump([par_tr, par_te], f)
    f.close()
    