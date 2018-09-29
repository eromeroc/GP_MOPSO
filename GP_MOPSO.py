############################################################################
# Autora: Elena Romero Contreras
# Ejecución del algoritmo de generación de prototipos usando MOPSO
# y clasificación del conjunto test usando k-NN sobre los prototipos obtenidos
#############################################################################

import sys
import random
import time
import MOPSO 
import numpy as np
import data as dt
import fitness as fit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, train_test_split



def GP_MOPSO(data=None, train=None, test=None, fitness=fit.fitness,
			n_fit = 3, n_rep=100, num_part=100, max_iter=200, c1=1, c2=2):

	"""
	Ejecución del algoritmo GP_MOPSO sobre un conjunto de datos (ya dividido en train y test o no)
	y posterior clasificación del conjunto test usando k-NN sobre los prototipos obtenidos
	"""
	if data:
		X_train, X_test, y_train, y_test = train_test_split(data.values, data.label, test_size=0.2, random_state=42)
		train = dt.Data(X= X_train, y= y_train)
		test = dt.Data(X= X_test, y= y_test)

	start = time.time()

	# Aplica MOPSO
	repX, repF = MOPSO.MOPSO(train, fitness, n_fit, n_rep, num_part, max_iter, c1, c2)
	# Nos quedamos con la mejor partícula (conjunto de prototipos)
	min_index = np.argmin(np.mean(repF, axis=1))
	prototypes = repX[min_index]

	end = time.time()

	#----------- TEST -----------
	
	## 1-NN ##
	knn = KNeighborsClassifier(n_neighbors=1) 
	knn.fit(prototypes, train.classes)
	pred_label = knn.predict(test.values)

	# Medición resultados
	n_success = sum(np.array(pred_label == test.label))

	accuracy = n_success/test.n_row
	t = end-start

	return (accuracy, t)



if __name__ =='__main__':

	fname = sys.argv[1]

	if len(sys.argv)>2:
		label_first = sys.argv[2]
	else:
		label_first = None

	#Fijamos semillas
	random.seed(17)
	np.random.seed(17)

	# Lectura de datos
	X = dt.Data(fname=fname, label_first = label_first)
	X.normalize()

	
	# 5-fold cross validation
	kf = KFold(n_splits = 5, shuffle=True, random_state= 42)
	#Inicializamos medidas
	total_accuracy = []
	total_t = []

	for train_index, test_index in kf.split(X.values):
		train = dt.Data(X= X.values[train_index], y= X.label[train_index])
		test = dt.Data(X= X.values[test_index], y= X.label[test_index])
		accuracy, t = GP_MOPSO(train=train, test=test)
		total_accuracy.append(accuracy)
		total_t.append(t)

	print(fname, "\t{0:.5f}".format(np.mean(total_accuracy)), "\t{0:.5f}".format(np.mean(t)))

