from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import distance_matrix
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import multiprocessing as mp
import tensorflow as tf
import numpy as np
import progressbar
import queue
import json

# Set the path to load the tfrecords and json files
load_path = './data_raw'
# Set the path to save the final np data
save_path =  './data'

# Antenna definitions

# Default DICHASUS version:
ASSIGNMENTS = [
	[0, 13, 31, 29, 3, 7, 1, 12 ],
	[30, 26, 21, 25, 24, 8, 22, 15],
	[28, 5, 10, 14, 6, 2, 16, 18],
	[19, 4, 23, 17, 20, 11, 9, 27]
]
# Edit the default DICHASUS version for nicely sorted antennas and arrays:
ASSIGNMENTS = [
    [6,2,16,18, 28,5,10,14],
    [24,8,22,15, 30,26,21,25],
    [3,7,1,12, 0,13,31,29],
    [20,11,9,27, 19,4,23,17]]

ANTENNACOUNT = np.sum([len(antennaArray) for antennaArray in ASSIGNMENTS])

def load_calibrate_timedomain(path, offset_path):
	offsets = None
	with open(offset_path, "r") as offsetfile:
		offsets = json.load(offsetfile)

	def record_parse_function(proto):
		record = tf.io.parse_single_example(
			proto,
			{
				"csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
				"pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
				"time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
				"snr": tf.io.FixedLenFeature([], tf.string, default_value = '') # sue
			},
		)

		csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (ANTENNACOUNT, 1024, 2))
		csi = tf.complex(csi[:, :, 0], csi[:, :, 1])
		csi = tf.signal.fftshift(csi, axes=1)

		position = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3))
		time = tf.ensure_shape(record["time"], ())

		snr = tf.ensure_shape(tf.io.parse_tensor(record["snr"], out_type = tf.float32), (32)) # sue

		return csi, position[:2], time, snr

	def apply_calibration(csi, pos, time, snr):
		sto_offset = tf.tensordot(tf.constant(offsets["sto"]), 2 * np.pi * tf.range(tf.shape(csi)[1], dtype = np.float32) / tf.cast(tf.shape(csi)[1], np.float32), axes = 0)
		cpo_offset = tf.tensordot(tf.constant(offsets["cpo"]), tf.ones(tf.shape(csi)[1], dtype = np.float32), axes = 0)
		csi = tf.multiply(csi, tf.exp(tf.complex(0.0, sto_offset + cpo_offset)))

		return csi, pos, time, snr

	def csi_time_domain(csi, pos, time, snr):
		csi = tf.signal.fftshift(tf.signal.ifft(tf.signal.fftshift(csi, axes=1)),axes=1)

		return csi, pos, time, snr

	def cut_out_taps(tap_start, tap_stop):
		def cut_out_taps_func(csi, pos, time, snr):
			return csi[:,tap_start:tap_stop], pos, time, snr

		return cut_out_taps_func

	def order_by_antenna_assignments(csi, pos, time, snr):
		csi = tf.stack([tf.gather(csi, antenna_inidces) for antenna_inidces in ASSIGNMENTS])
		snr = tf.stack([tf.gather(snr, antenna_inidces) for antenna_inidces in ASSIGNMENTS]) # sue
		return csi, pos, time, snr

	dataset = tf.data.TFRecordDataset(path)

	dataset = dataset.map(record_parse_function, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(apply_calibration, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(csi_time_domain, num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(cut_out_taps(507, 520), num_parallel_calls = tf.data.AUTOTUNE)
	dataset = dataset.map(order_by_antenna_assignments, num_parallel_calls = tf.data.AUTOTUNE)

	return dataset

inputpaths = [
	{
		"tfrecords" : f"{load_path}/dichasus-cf02.tfrecords",
		"offsets" : f"{load_path}/reftx-offsets-dichasus-cf02.json"
	},
	{
		"tfrecords" : f"{load_path}/dichasus-cf03.tfrecords",
		"offsets" : f"{load_path}/reftx-offsets-dichasus-cf03.json"
	}
]

full_dataset = load_calibrate_timedomain(inputpaths[0]["tfrecords"], inputpaths[0]["offsets"])

for path in inputpaths[1:]:
	full_dataset = full_dataset.concatenate(load_calibrate_timedomain(path["tfrecords"], path["offsets"]))

# Default DICHASUS code decimates the dataset, uses only every 4th datapoint (to reduce number of points)
# training_set = full_dataset.enumerate().filter(lambda idx, value : (idx % 4 == 0))

# Keep the whole dataset instead:
training_set = full_dataset.enumerate().filter(lambda idx, value : (idx % 1 == 0))
training_set = training_set.map(lambda idx, value : value)

groundtruth_positions = []
csi_time_domain = []
timestamps = []

for csi, pos, time, snr in training_set.batch(1000):
	csi_time_domain.append(csi.numpy())
	groundtruth_positions.append(pos.numpy())
	timestamps.append(time.numpy())

csi_time_domain = np.concatenate(csi_time_domain)
groundtruth_positions = np.concatenate(groundtruth_positions)
timestamps = np.concatenate(timestamps)

# Number of samples of each tfrecords file (could be useful to single out segments later)
n_ele_per_seg = np.array([18516, 23478]) # num of samples in cf02 and cf03
segment_start_idcs = np.cumsum(np.concatenate((np.array([0]), n_ele_per_seg)))

# Sort each segment in ascending order of timestamps
for seg_idx, idx in enumerate(segment_start_idcs[:-1]):
  tmp = np.arange(idx, segment_start_idcs[seg_idx+1], dtype=int)
  sorting = np.argsort(timestamps[tmp])
  csi_time_domain[tmp] = csi_time_domain[tmp[sorting]]
  groundtruth_positions[tmp] = groundtruth_positions[tmp[sorting]]
  timestamps[tmp] = timestamps[tmp[sorting]]

print(csi_time_domain.shape)
print(groundtruth_positions.shape)
print(timestamps.shape)
# Done with all np variables to save!

csi_np = np.transpose(csi_time_domain, (0, 3, 1, 2))
csi_np = np.reshape(csi_np, (csi_np.shape[0], csi_np.shape[1], -1))
csi_np = np.transpose(csi_np, (0, 2, 1))
# csi_np is used for per-UE features, make sure that the dimensions are correct:
print(csi_np.shape)

def plot_colorized(positions, groundtruth_positions, title = None, show = True, alpha = 1.0):
	# Generate RGB colors for datapoints
	center_point = np.zeros(2, dtype = np.float32)
	center_point[0] = 0.5 * (np.min(groundtruth_positions[:, 0], axis = 0) + np.max(groundtruth_positions[:, 0], axis = 0))
	center_point[1] = 0.5 * (np.min(groundtruth_positions[:, 1], axis = 0) + np.max(groundtruth_positions[:, 1], axis = 0))
	NormalizeData = lambda in_data : (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))
	rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
	rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
	rgb_values[:, 1] = 0.8 * NormalizeData(np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1)))
	rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

	# Plot datapoints
	plt.figure(figsize=(6, 6))
	if title is not None:
		plt.title(title, fontsize=16)
	plt.scatter(positions[:, 0], positions[:, 1], c = rgb_values, alpha = alpha, s = 10, linewidths = 0)
	plt.xlabel("x coordinate")
	plt.ylabel("y coordinate")
	if show:
		plt.show()

plot_colorized(groundtruth_positions, groundtruth_positions, title="Ground Truth Positions")

np.save(f'{save_path}/csi_calibrated_13taps_per_ap.npy', csi_time_domain)
# csi_np can always be recomputed whenever csi_time_domain_per_ap is loaded, no need to save it separately
# np.save(f'{save_path}/csi_calibrated_13taps.npy', csi_np)
np.save(f'{save_path}/pos_tachy.npy', groundtruth_positions)
np.save(f'{save_path}/time.npy', timestamps)
np.save(f'{save_path}/n_segment.npy', n_ele_per_seg)

