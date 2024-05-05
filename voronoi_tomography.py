import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def parse_ore(path_ore):
	"""
	Returns a list containing the coordinates of each copper mine.

	The filepath should point to the file path containing the geojson of
	the ore data. This can be converted using from the orginal ore.gml data
	into an geojson using QGIS or other means.

	We read the geojson into a dictionary and extract the coordinates which are
	contained in the features. Then we can include only Copper (Cu) mines and
	append the coordinates to a list and return as a numpy array.

	path_ore: path to ore data
	"""

	ore_data = None
	with open(path_ore, "r", encoding="utf-8") as f:
		ore_data = json.load(f)['features']

	filtered_list = []
	for point in ore_data:
		if 'Cu' in point['properties']['GRUPO_RECU']:
			filtered_list.append(np.array(point['geometry']['coordinates']))
	return np.array(filtered_list)

def parse_tomo(path_vp, path_vs, path_vpvs):
	"""
	Returns a list containing the coordinates and each feature from the
	tomography data.

	The three paths should point to the respective tomography data. There should
	be 3 different files for this data, Vp (compressional), Vs (shear), and the
	ratio, VpVs.

	We read each csv file and append each datapoint along their columns. This is
	done because each measurement across Vp, Vs, and VpVs has their own
	corresponding measurement in the other files. Then for all 32 measurements
	at a certain coordinate, we append them into one feature vector along with
	the coordinate and store it as a list (shape=(n by 98)).

	path_vp: path to Vp data
	path_vs: path to Vs data
	path_vpvs: path to VpVs data
	"""

	vp = pd.read_csv(path_vp, delimiter=';')
	vs = pd.read_csv(path_vs, delimiter=';')
	vpvs = pd.read_csv(path_vpvs, delimiter=';')

	tomo_df = pd.concat([vp['este (x) utm'],vp['norte (y) utm'],vp['Altura km'],vp['z'],vp['Vp'],vs['Vs'],vpvs['VpVs']],axis=1)
	tomo_df.columns = ["X","Y","Height","z","Vp","Vs","VpVs"]

	tomography = []

	unique = np.unique(tomo_df['X'].to_numpy())
	for X in unique:
		rows = tomo_df.loc[tomo_df['X']==X]
		coord = rows.iloc[:1,:2].to_numpy()[0]
		# features = rows[['Vp','Vs','VpVs']].to_numpy().flatten()
		# features = rows[['Vp']].to_numpy().flatten()
		features = rows[['Vs']].to_numpy().flatten()
		# features = rows[['Vp','Vs']].to_numpy().flatten()

		data = np.concatenate((coord,features))
		tomography.append(data)

	return np.array(tomography)

def get_mine_tomo_points(mines, tomography):
	"""
	Returns a list containing the coordinates of each tomography point matched
	with a mine point.

	For each mine in our list of copper mine location, we calculate the euclidean
	distnace between the mine and tomograph point UTM coordinates. We then take the
	minimum distance and append the tomography coordinate to our list. Finally,
	duplicate coordinates are removed as the sampling size of the tomography
	data is quite large so a single tomogrpahy data point can "contain" multiple
	mines.

	mines: copper mine coordinates
	tomography: tomography data
	"""

	matched_points = []
	for mine in mines:
		tomo_dist = np.linalg.norm(mine-tomography[:,:2],axis=1)
		min_idx = np.argmin(tomo_dist)
		matched_points.append(tomography[min_idx,:2])
	return np.unique(np.array(matched_points),axis=0)

def generate_voronoi(points, sample=False, n=15, drawing=False):
	"""
	Returns vornoi graph generated from tomograph points matched with the mine
	coodinates.

	If we choose to sample, we randomly take n number of coordinate pairs
	from our matched point list. If we do not, we generate a voronoi graph
	using all matched points. To do the graph generation, we use scipy's
	implementation.

	points: Matched tomography/mine points
	sample: Control parameter for using subset of points
	n: Number of partitions (default is # of points)
	drawing: Control parameter for outputing visualization of the voronoi graph
	"""

	sample_idx = None
	if sample:
		sample_idx = np.random.choice(len(matched_points),n,replace=False)
		points = points[sample_idx]
	else:
		sample_idx = np.arange(len(points))

	vor = Voronoi(matched_points[sample_idx])

	if drawing:
		fig = voronoi_plot_2d(vor)
		plt.show()

	return vor, sample_idx

def waxman_prob(points, vor, tomography, alpha=0.1, beta=0.4):
	"""
	Uses the non-sampled mines to calculate the probability of an
	edge between the Voronoi partition center and each non-sampled
	mine points. This can be influenced using alpha and beta parameters.

	For each point not used in the voronoi graph creation, we match
	the point to the partition they are located in. We then calculate
	the feature distance between the mine and the center mine. From this
	value, we can calculate our probability using b*exp(-d/a).

	points: non-sampled mine points
	vor: voronoi graph
	tomography: tomography data
	alpha: Alpha value
	beta: Beta value
	"""
	centers = vor.points
	dist_dict = {}
	for i in range(len(centers)):
		dist_dict[i] = []

	# Match points to partitions
	for p in points:
		dist = np.linalg.norm(p-centers,axis=1)
		partition_idx = np.argmin(dist)

		mask1 = (tomography[:,0]==centers[partition_idx][0])
		mask2 = (tomography[:,0]==p[0])
		center_point = tomography[mask1,:][0]
		feat_point = tomography[mask2,:][0]
		f_dist = np.linalg.norm(center_point[2:]-feat_point[2:])
		dist_dict[partition_idx].append(f_dist)

	prob_list = []
	for i in range(len(dist_dict)):
		vals = np.array(dist_dict[i])
		L = max(vals)-min(vals)
		probs = beta*np.exp(-1*vals/(alpha*L))
		avg_prob = np.average(probs)
		# print("Partition {}: Highest average probability of {} with a/b of {}".format(i,best_prob,best_params))
		prob_list.append(avg_prob)
	print("Average partition accuracy:", np.average(np.array(prob_list)))

def calc_dist_feature(tomography, points, vor):
	"""
	Calculates the feature distance between all points 
	within a partition to its center. This also labels
	points with 0 (non-mine) or 1 (mine) for use in
	classification.

	For each tomography data point, we find the partition
	it belongs to and calculate the feature distance from
	the center to the point. We also calculate the physical
	distance using the coordinates as another feature. We
	then check if the tomography point is a mine point and
	label it accordingly.

	tomography: tomography data
	points: non-sampled mine points
	vor: voronoi graph
	"""

	centers = vor.points
	dist_dict = {}
	for i in range(len(centers)):
		dist_dict[i] = []

	for tomo in tomography:
		dist = np.linalg.norm(tomo[:2]-centers,axis=1)
		partition_idx = np.argmin(dist)
		if min(dist)==0:
			continue

		mask = (tomography[:,0]==centers[partition_idx][0])
		center_point = tomography[mask,:][0]

		f_dist = np.linalg.norm(tomo[2:]-center_point[2:])
		p_dist = np.linalg.norm(tomo[:2]-center_point[:2])
		if tomo[:2] in points:
			dist_dict[partition_idx].append([f_dist, p_dist, 1])
		else:
			dist_dict[partition_idx].append([f_dist, p_dist, 0])
		
	return dist_dict

def logistic_regression(data, threshold=.5):
	"""
	Uses sklearn's logistic regression to fit a model
	and evaluate it for each partition.

	For each partition, we extract the feature distances
	and the labels as X and y. We also balance the dataset
	as there are a lot more negative labels than positive.
	From there, we split the data into training and testing
	sets using a 80/20 split. We then fit the model and report
	both training and testing accuracies.

	data: Feature distance by partition
	threshold: Optional parameter for classification threshold
	"""

	avg_train_acc = 0
	avg_test_acc = 0

	for i in range(len(data)):
		values = np.array(data[i])
		# Feature distance
		X = values[:,0].reshape(-1,1)
		# Feature and physical distance
		# X = values[:,:2]
		y = values[:,2]

		# Balance the dataset to be 50/50 between non-mines and mines
		neg_idx = np.where(y == 0)[0]
		pos_idx = np.where(y == 1)[0]
		rm_idx = np.random.choice(neg_idx, size=(len(neg_idx)-len(pos_idx)), replace=False)
		X = np.delete(X, rm_idx, axis=0)
		y = np.delete(y, rm_idx)

		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

		clf = LogisticRegression().fit(X_train, y_train)
		train_preds = np.where(clf.predict_proba(X_train)[:,1] > threshold, 1, 0)
		train_acc = accuracy_score(y_train,train_preds)

		# print("Training Score",train_acc)

		test_preds = np.where(clf.predict_proba(X_test)[:,1] > threshold, 1, 0)
		test_acc = accuracy_score(y_test,test_preds)
		# print("Testing Score",test_acc)
		# cm = confusion_matrix(y_test, preds)
		# print(cm)
		avg_train_acc += train_acc
		avg_test_acc += test_acc
	print("Average training accuracy:",avg_train_acc/len(data))
	print("Average testing accuracy:",avg_test_acc/len(data))

def logistic_regression2(data, threshold=.5):
	"""
	Uses sklearn's logistic regression to fit a model
	and evaluate it over the aggregate of every partition.

	We first aggregate the data from each partition as one
	dataset. We then use this dataset to extract our features
	and the labels as X and y. We also balance the dataset
	as there are a lot more negative labels than positive.
	From there, we split the data into training and testing
	sets using a 80/20 split. We then fit the model and report
	both training and testing accuracies.

	data: Feature distance by partition
	threshold: Optional parameter for classification threshold
	"""

	data = list(data.values())
	values = []
	for d in data:
		values = values+d
	values = np.array(values)
	# Feature distance
	X = values[:,0].reshape(-1,1)
	# Feature and physical distance
	# X = values[:,:2]
	y = values[:,2]

	# Balance the dataset to be 50/50 between non-mines and mines
	neg_idx = np.where(y == 0)[0]
	pos_idx = np.where(y == 1)[0]
	rm_idx = np.random.choice(neg_idx, size=(len(neg_idx)-len(pos_idx)), replace=False)
	X = np.delete(X, rm_idx, axis=0)
	y = np.delete(y, rm_idx)

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)
	clf = LogisticRegression().fit(X_train, y_train)

	train_preds = np.where(clf.predict_proba(X_train)[:,1] > threshold, 1, 0)
	train_acc = accuracy_score(y_train,train_preds)
	print("Training Score",train_acc)

	preds = np.where(clf.predict_proba(X_test)[:,1] > threshold, 1, 0)
	print("Testing Score",accuracy_score(y_test,preds))
	cm = confusion_matrix(y_test, preds)
	# print(cm)

def deep_nn(data, ratio=1):
	"""
	Uses a 3 hidden layer deep neural network to predict
	mine probabilities based on the feature distances
	generated by the Voronoi partitions.

	We first extract the features and the labels into our
	X and y before balancing the data. To balance this
	data, we have a ratio (default 1) of positive to negative
	data. We then create our neural network using 3 hidden
	layers: 12, 8, and 2 nodes using relu for the first two
	and softmax for the last. This allows us to output the
	probability of a point being a mine. We then print out
	the training and testing accuracy.

	data: Feature distance by partition
	ratio: Ratio of positive to negative labels (1:ratio)
	"""

	data = list(data.values())
	values = []
	for d in data:
		values = values+d
	values = np.array(values)

	X = values[:,0].reshape(-1,1)
	y = values[:,2]

	neg_idx = np.where(y == 0)[0]
	pos_idx = np.where(y == 1)[0]
	rm_idx = np.random.choice(neg_idx, size=(len(neg_idx)-round(ratio*len(pos_idx))), replace=False)

	y = to_categorical(y, num_classes=2)
	X = np.delete(X, rm_idx, axis=0)
	y = np.delete(y, rm_idx, axis=0)

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)

	model = Sequential()
	model.add(Dense(12, input_shape=(1,), activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(2,activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	model.fit(X_train,y_train,epochs=100,batch_size=10,verbose=0)

	_, accuracy = model.evaluate(X_train, y_train,verbose=0)
	print('Training Accuracy: %.2f' % (accuracy*100))
	_, accuracy = model.evaluate(X_test, y_test,verbose=0)
	print('Testing Accuracy: %.2f' % (accuracy*100))
	return model

def generate_isomap(tomography, vor, model, drawing=False):
	"""
	Creates an isomap using the neural network to determine
	mine probabilities from every point.

	For every feature point, we find the partition it
	belongs to and calculate the feature distance between the
	point and the Vornoi partition point. We then aggregate
	all feature distances and their coordinates into one dataset
	to pass into our neural network. This returns the probability
	a point is a mine which we can relate with a coordinate.

	tomography: Tomography data
	vor: Voronoi graph
	model: Keras neural network
	drawing: Control parameter for plotting isomap
	"""

	centers = vor.points

	dist_dict = {}
	for i in range(len(centers)):
		dist_dict[i] = []

	for tomo in tomography:
		dist = np.linalg.norm(tomo[:2]-centers,axis=1)
		partition_idx = np.argmin(dist)
		mask = (tomography[:,0]==centers[partition_idx][0])
		center_point = tomography[mask,:][0]
		if min(dist)==0:
			continue

		f_dist = np.linalg.norm(tomo[2:]-center_point[2:])
		dist_dict[partition_idx].append([tomo[0],tomo[1],f_dist])

	all_X = []
	all_Y = []
	all_probs = []
	for i in range(len(dist_dict)):
		values = np.array(dist_dict[i])
		X = values[:,0]
		Y = values[:,1]
		features = values[:,2]
		mine_probs = model.predict(features,verbose=0)[:,1]
		all_X = all_X+X.tolist()
		all_Y = all_Y+Y.tolist()
		all_probs = all_probs+mine_probs.tolist()

	if drawing:
		colors = []
		for prob in all_probs:
			if prob < .33:
				colors.append('red')
			elif prob < .66:
				colors.append('green')
			else:
				colors.append('blue')
		plt.scatter(all_X,all_Y,c=colors)
		plt.scatter(centers[:,0],centers[:,1],c='black')
		plt.show()

if __name__ == "__main__":
	# Testing and reproducibility
	np.set_printoptions(suppress=True,precision=3)
	np.random.seed(100)

	# Retrieve copper mine data
	filepath_mine = 'ore_body.geojson'
	mines = parse_ore(filepath_mine)

	# Retreive tomography data
	filepath_vs = 'tomography/JOINT4-mod10-Vs-UTM.csv'
	filepath_vp = 'tomography/JOINT4-mod10-Vp-UTM.csv'
	filepath_vpvs = 'tomography/JOINT4-mod10-VpVs-UTM.csv'
	tomography = parse_tomo(filepath_vp, filepath_vs, filepath_vpvs)

	# Retrieve mine points corresponding to tomography points
	matched_points = get_mine_tomo_points(mines, tomography)
	# Generate vornoi graph from matched points
	vor, sample_idx = generate_voronoi(matched_points, sample=True, n=20, drawing=True)

	# Get non-sampled matched points
	mask = np.ones(len(matched_points),dtype=bool)
	mask[sample_idx] = 0
	non_sampled = matched_points[mask]
	
	########################################
	#### Method 1: Waxman Probabilities ####
	########################################

	waxman_prob(non_sampled, vor, tomography)

	########################################
	#### Method 2: Logistic Regression #####
	########################################

	dist_data = calc_dist_feature(tomography, non_sampled, vor)
	logistic_regression(dist_data, threshold=.5)
	logistic_regression2(dist_data, threshold=.5)

	########################################
	#### Method 3: Deep Neural Network #####
	########################################

	nn = deep_nn(dist_data, ratio=2)

	# Calculate ISOMAP for each Voronoi partition
	generate_isomap(tomography, vor, nn, True)



	

	