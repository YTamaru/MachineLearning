# algorithm
# 画像を入力
# 全画像をCNN（AlexNetなど）にかけて、局所特徴量の抽出
# 全局所特徴量をk-meansによってカテゴライズして、ヒストグラムのビンとする
# ビンをもとに、全画像に対してVisualWordsでヒストグラムを作り、特徴ベクトルを作成
# 全特徴ベクトルに対して、k-NN グラフを作成し、cycle consistency によってカテゴリを作成する
# カテゴリ数の表示及び、同一シーンの視点変化させた画像群に対して、同じカテゴリに属しているかの判断を行う


# coding: utf-8

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import sys
import cv2
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt

DATA_DIR = '../data/'
VIDEOS_DIR = '../data/video/'                        # The place to put the video
TARGET_IMAGES_DIR = '../data/images/target/'         # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '../data/images/clustered/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label

num_img = 285
DG = nx.DiGraph()

class Image_Clustering:
	def __init__(self, n_clusters=50, video_file='IMG_2140.MOV', image_file_temp='img_%s.png', input_video=False, col=7*7, channels=512, k_NN=4):
		self.n_clusters = n_clusters            # The number of cluster
		self.video_file = video_file            # Input video file name
		self.image_file_temp = image_file_temp  # Image file name template
		self.input_video = input_video          # If input data is a video
		self.col = col
		self.channels = channels
		self.k_NN = k_NN

	def main(self):
		if self.input_video == True:
			self.video_2_frames()
		self.feat_images() 

	def video_2_frames(self):
		print('Video to frames...')
		cap = cv2.VideoCapture(VIDEOS_DIR+self.video_file)

		# Remove and make a directory.
		if os.path.exists(TARGET_IMAGES_DIR):
			shutil.rmtree(TARGET_IMAGES_DIR)  # Delete an entire directory tree
		if not os.path.exists(TARGET_IMAGES_DIR):
			os.makedirs(TARGET_IMAGES_DIR)	# Make a directory

		i = 0
		global num_img
		while(cap.isOpened()):
			flag, frame = cap.read()  # Capture frame-by-frame
			if flag == False:
				break  # A frame is not left
			cv2.imwrite(TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6), frame)  # Save a frame
			i += 1
			print('Save', TARGET_IMAGES_DIR+self.image_file_temp % str(i).zfill(6))
		num_img = i
		cap.release()  # When everything done, release the capture
		print('')


	def feat_images(self):
		print('Label images...')

		# Load a model
		model = VGG16(weights='imagenet', include_top=False)
	
		# Get images
		images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.png', '.jpg']]
		assert(len(images)>0)
		
		X = []
		X_vector = []
		for i in range(len(images)):
			# Extract image features
			feat = self.__feature_extraction(model, TARGET_IMAGES_DIR+images[i])
			X.append(feat)
		# Clutering images by k-means++
		X = np.array(X)
		X = X.reshape([num_img*self.col, self.channels])
		kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(X)
		centr = np.array(kmeans.cluster_centers_)


		#各画像の特徴ベクトルの作成 Bag of Visual Words
		for i in range(num_img-1):
			X_hist = [0]*self.col
			for j in range(self.col-1):
				dist = [0]*self.col
				for k in range(self.n_clusters-1):
					diff = X[i*self.col+j,] - centr[k,]
					dist[k] = np.linalg.norm(diff)
				X_hist[dist.index(min(dist))] += 1
			X_vector.append(X_hist)
		X_vector = np.array(X_vector) #feature vector
		print(np.shape(X_vector)) #(284,49)? (285,49)
		DG.add_nodes_from(range(1, len(X_vector)))

		# #k-NNグラフの作成(初期値は最初の画像、サイクルのノードは削除していき、次の初期値は画像のインデックスの小さい方から選択する)
		# cycle = []
		# X_list = []
		# for i in range(1, len(X_vector)):	
		# 	X_list.append(i)
		# while len(X_vector)!=0:
		# 	node_I = []
		# 	for i in range(len(X_vector)-1):
		# 		L2 = []
		# 		for j in range(len(X_vector)-1):
		# 			if i != j:
		# 				diff = X_vector[i,] - X_vector[j,]
		# 				L2.append(np.linalg.norm(diff)) #L2norm
		# 		L2 = sorted(range(len(L2)), key=lambda k: L2[k])[:self.k_NN] #リストの値を昇順に並び替えて元の要素のインデックスの値で返す 上位ｋ個→ダメ
		# 		if L2[0] != node_I[0]:
		# 			node_I.append(L2[0])
		# 		else:
		# 			break
		# 	cycle.append(node_I)
		# 	for l in range(len(node_I)):
		# 		np.delete(X_vector, node_I[l], 0)

		# ##元のインデックスを保持したまま、要素を削除
		# cycle = np.array(cycle)
		# print(cycle)
		# print(np.shape(cycle))

		# #cycleごとにimageを分類


		cycle = []
		X_list = X_vector
		np.array(X_list)
		lbl = []
		for i in range(1, len(X_list)+1):
			lbl.append([i])
		X_list = np.append(X_list,lbl,axis=1) # X_list = [,,,,,img_num]
		while len(X_list)>1:
			node_I = []
			for i in range(len(X_list)-1):
				node_I.append(X_list[0,self.col])
				L2 = []
				for j in range(len(X_list)-1):
					if i!=j:
						diff = X_list[i,] - X_list[j,]
						np.delete(diff, len(diff)-1,0)
						L2.append(np.array([i,j,np.linalg.norm(diff)])) #L2norm [I, kNN,L2norm]
				L2 = np.array(L2)
				L2 = L2[np.argsort(L2[:,2])]
				L2 = np.delete(L2, slice(self.k_NN, len(L2)),0)
				if L2[0,1] != node_I[0]:
					node_I.append(L2[0,1])
					print(node_I)
					# for m in range(self.k_NN-1):
						# DG.add_edges_from([(L2[0,0],L2[m,2])])
				else:
					break
			cycle.append(node_I)
			for l in range(len(node_I)):
				for n in range(len(X_list)-1):
					if node_I[l] == X_list[n,self.col]:
						X_list = np.delete(X_list, n, 0)

		# plt.subplot(121)
		# nx.draw(DG, with_labels=True, font_weight='bold')
		# plt.subplot(122)
		# nx.draw_shell(DG, nlist=[range(5,10), range(5)], with_labels=True, font_weight='bold')
		# plt.show()

		print(cycle)


	def __feature_extraction(self, model, img_path):
		img = image.load_img(img_path, target_size=(224, 224))  # resize
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)  # add a dimention of samples
		x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels

		feat = model.predict(x)  # Get image features #1次元化しないで7x7x512

		return feat

		
if __name__ == "__main__":
	Image_Clustering().main()