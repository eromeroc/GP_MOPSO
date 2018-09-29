#######################################################
# Author: Elena Romero Contreras
# Calculate Error Rank
#######################################################

from sklearn.neighbors import KNeighborsClassifier
from scipy.special import expit
import numpy as np
import itertools
import time


def fitness(data, prototypes, n_fit, parts):
	"""
	Calcula error rank para cada subconjunto en los que se divide el conjunto de datos
	"""

	# Calculamos error rank para cada subconjunto
	F = []
	for i in range(n_fit):
		F.append(calculateTotalErrorRank(prototypes, data.classes, data.values[parts[i]], data.label[parts[i]]))

	return np.array(F)



def calculateTotalErrorRank(prototypes, classes, values, label):
	"""
	Calcula el error rank total sumando error rank de cada problema binario
	"""
	if(len(classes) == 2): # Si ya es un problema binario
		total_error = calculateErrorRank(prototypes, classes, values, label)
	else:
		pairs = itertools.combinations(classes, 2)	# Obtenemos pares de clases (one vs one)
		total_error = 0
		# Para cada par de clases calculamos error rank
		for i,j in pairs:
			# Nos quedamos con los prototipos de las dos clases
			index_i = classes.index(i)
			index_j = classes.index(j)
			bin_prototypes = [prototypes[index_i,:], prototypes[index_j,:]]
			
			# Nos quedamos con las instancias de las dos clases
			bin_label = []
			bin_values = []
			for l in range(len(label)):
				if (label[l] == i or label[l] == j):
					bin_label.append(label[l])
					bin_values.append(values[l,:])

			# Si hay instancias de esa clase, calculamos error rank
			if bin_label:
				total_error += calculateErrorRank(bin_prototypes, [i,j], bin_values, bin_label)

	return total_error


def calculateErrorRank(prototypes, classes, values, label):
	"""
	Método que calcula el rango de error
	"""
	N = len(label)
	d = calculateDiscriminant(prototypes, classes, values)
	pred_label = []

	for i in d:
		if i<0:
			pred_label.append(classes[0])
		else:
			pred_label.append(classes[1])


	# Ordenamos las instancias por el valor del discriminante

	zipped = zip(d, label, pred_label)
	s = sorted(zipped,  key=lambda x: x[0])
	d, label, pred_label = zip(*s) # Unzip

	#Primer elemento no negativo
	z = firstNoNeg(d)
	r = []

	for i in range(N):
		if i < z:
			r.append((i-z)/z)
		else:
			r.append((i-z)/(N-z))

		r[i] = expit(r[i])

	error = 0
	for i in range(N):
		if label[i] != pred_label[i]:
			error += abs(r[i])

	return error




def calculateErrorRate(label, pred_label):
	"""
	Calcula el ratio de error: proporción de instancias mal clasificadas
	"""
	error = sum(np.array(label != pred_label)) / len(label)

	return error




def calculateDiscriminant(prototypes, classes, values):
	d = []
	c1 = classes[0] 
	c2 = classes[1]
	bool_c1 = False
	bool_c2 = False


	knn = KNeighborsClassifier(n_neighbors=len(classes))
	knn.fit(prototypes, classes)

	for i in values:
		dist, kn_index = knn.kneighbors(X=[i], return_distance=True)
		dist = dist[0]
		kn_index = kn_index[0]

		for j in range(len(kn_index)):
			if ((classes[kn_index[j]] == c1) and not bool_c1):
				p_c1 = kn_index[j]
				dist_c1 = dist[j]
				bool_c1 = True
			elif((classes[kn_index[j]] == c2) and not bool_c2):
				p_c2 = kn_index[j]
				dist_c2 = dist[j]
				bool_c2 = True

		d.append(dist_c1 - dist_c2)
		bool_c1=False
		bool_c2=False

	return d




def firstNoNeg(vector):
	"""
	Devuelve el primer elemento no negativo
	"""
	for i in vector:
		if(i>=0):
			return i

	return (len(vector)-1)




