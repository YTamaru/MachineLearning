import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

DG = nx.DiGraph()
DG.add_nodes_from(range(1, len(X_vector)))


#k_NN
cycle = []
X_list = X_vector
np.array(X_list)
for i in range(1, len(X_vector)):
    X_list[i,].append(i) # X_list = [,,,,,img_num]
while len(X_list)!=0:
    node_I = []
    for i in range(len(X_list)-1):
        node_I.append(X_list[0,self.col])
        L2 = []
        for j in range(len(X_list)-1):
            if i!=j:
                diff = X_list[i,] - X_list[j,]
                diff.pop(len(X_list)-1)
                L2.append([i,j,np.linalg.norm(diff)]) #L2norm [I, kNN,L2norm]
        L2 = np.array(L2)
        L2 = L2[np.argsort(L2[:,2])]
        np.delete(L2, slice(self.k_NN, len(L2)),0)
        if L2[0,2] != node_I[0]:
            node_I.append(L2[0,2])
            for m in range(self.k_NN-1):
                DG.add_edges_from(L2[0,0],L2[m,2])
        else:
            break
    cycle.append(node_I)
    for l in range(len(node_I)):
        for n in range(len(X_list)-1):
            if node_I[l] == X_list[n,self.col]:
                np.delete(X_list, n, 0)

plt.subplot(121)
nx.draw(DG, with_labels=True, font_weight='bold')
plt.subplot(122)
nx.draw_shell(DG, nlist=[range(5,10), range(5)], with_labels=True, font_weight='bold')
plt.show()

print[cycle]







		#k-NNグラフの作成(初期値は最初の画像、サイクルのノードは削除していき、次の初期値は画像のインデックスの小さい方から選択する)
		cycle = []
		X_list = []
		for i in range(1, len(X_vector)):	
			X_list.append(i)
		while len(X_vector)!=0:
			node_I = []
			for i in range(len(X_vector)-1):
				L2 = []
				for j in range(len(X_vector)-1):
					if i != j:
						diff = X_vector[i,] - X_vector[j,]
						L2.append(np.linalg.norm(diff)) #L2norm
				L2 = sorted(range(len(L2)), key=lambda k: L2[k])[:self.k_NN] #リストの値を昇順に並び替えて元の要素のインデックスの値で返す 上位ｋ個→ダメ
				if L2[0] != node_I[0]:
					node_I.append(L2[0])
				else:
					break
			cycle.append(node_I)
			for l in range(len(node_I)):
				np.delete(X_vector, node_I[l], 0)

		##元のインデックスを保持したまま、要素を削除
		cycle = np.array(cycle)
		print(cycle)
		print(np.shape(cycle))

		#cycleごとにimageを分類
