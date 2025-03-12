# Channel Charting for Streaming CSI


This is the code for the results in the paper
"Channel Charting for Streaming CSI Data", S. Taner, M. Guillaud, O. Tirkkonen, and C. Studer
(c) 2025 Sueda Taner

email: taners@ethz.ch

### Important Information

If you are using this code (or parts of it) for a publication, then you _must_ cite the following paper:

S. Taner, M. Guillaud, O. Tirkkonen and C. Studer, "Channel Charting for Streaming CSI Data," Asilomar Conference on Signals, Systems, and Computers, 2023, pp. 1648-1653.

### How to use this code...

#### Step 1: Download and save the data

- From [dichasus-cf0x Dataset: Distributed Antenna Setup in Industrial Environment, Day 1](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/), download the ```dichasus-cf02``` and ```dichasus-cf03``` files (which should have the ```.tfrecord``` extension) and their offset estimates (which should be ```reftx-offsets-dichasus-cf02.json``` and ```reftx-offsets-dichasus-cf03.json```) into a folder called ```data_raw``.
- Run ```preprocess_dichasus.py```. This will store ```.np``` versions of the CSI, timestamps, and ground-truth positions extracted from the ```.tfrecord``` files in a folder called ```data```.

#### Step 2: Core memory curation for streaming CSI and channel charting

- Set your training and core memory curation parameters as explained on top of ```main.py``` and run for the results in the paper. This code does the following:
  - We use the ```cf02``` dataset to simulate the streaming CSI.
  - We store a subset of this dataset in the core memory.
  - We train a neural network for channel charting using the CSI features in the core memory.
  - We test the channel charting neural network on the ```cf03``` dataset.


### Version history

Version 0.1: taners@ethz.ch - initial version for GitHub release.

## Acknowledgments
This project makes use of the following external data and code:
- [dichasus-cf0x Dataset: Distributed Antenna Setup in Industrial Environment, Day 1](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/), accessed on 1/3/2025: Our code uses the ```cf02``` and ```cf03``` datasets.
- [Dissimilarity Metric-Based Channel Charting](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/) by F. Euchner, accessed on 1/3/2025: We use this code for (i) pre-processing the data from  ```tfrecords``` and ```json``` files, and (ii) computing the angle delay profile (ADP)-based distance metric and then the geodesic distances for training the channel charting network.
