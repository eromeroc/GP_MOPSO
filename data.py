#######################################################
# Autora: Elena Romero Contreras
# Clase Data con métodos para obtener prototipos
#######################################################


import numpy as np
import random
import itertools
from sklearn import preprocessing

class Data:
	"""
	Clase que representa un conjunto de datos donde cada fila es una instancia y
	cada columna una característica.
	Tiene métodos para leer, normalizar, calcular nº de prototipos y obtener prototipos
	"""

	def __init__(self, fname=None, X=None, y=None, label_first=None):

		"""
		Constructor de la clase Data
		Lee los datos de un fichero dado
		o crea conjunto de datos a partir de valores y etiquetas dadas
		"""
		if fname:
			self.values = []		# Matriz de instancias
			self.label = []			# Vector de etiquetas

			file = open(fname, 'r').read().splitlines()

			for line in file:
				current_line=line.split(",")
				
				if label_first:	#Si la etiqueta está al principio de línea
					self.label.append(current_line.pop(0))
				else:
					self.label.append(current_line.pop())

				v_data = []
				for i in current_line:
					if i=='?':
						v_data.append(0)
					else:
						v_data.append(float(i))

				self.values.append(np.array(v_data))

			self.values = np.array(self.values)
			self.label = np.array(self.label)
		else:
			self.values = X
			self.label = y
		
		self.n_row = len(self.values)
		self.n_col = len(self.values[0])
		self.classes = list(set(self.label))
		self.n_class = len(self.classes)	
		self.n_prot = self.getNumPrototypes()
		self.setClassesPairs()


	def getNumPrototypes(self, red_rate=0.99):
		"""
		Calcula el nº de prototipos por clase 
		Parámetros: red_rate -> ratio de reducción
		Devuelve vector con el nº de prototipos para cada clase
		"""
		n_prot = []
		for i in range(self.n_class):
			Nc = list(self.label).count(self.label[i])
			n_prot.append(max(1, round((1-red_rate)*Nc)))
		
		return n_prot

	def getClassIndexes(self, c):
		"""
		Devuelve todos los índices de las instancias con clase c
		"""
		indexes = []
		for i in range(self.n_row):
			if self.label[i] == c:
				indexes.append(i)

		return indexes

	def getPrototypesIndexes(self):
		"""
		Devuelve un vector con tantos indices aleatorios de instancias de cada clase
		como nº de prototipos corresponden a esa clase
		"""
		prot_indexes = []
		for i in range(self.n_class):
			indexes = self.getClassIndexes(self.classes[i]) #indices de las instancias pertenecientes a esa clase
			r_indexes = random.sample(indexes, self.n_prot[i])

			for j in range(self.n_prot[i]):
				prot_indexes.append(r_indexes[j])

		return prot_indexes


	def merge(self, indexes):
		"""
		Mezcla las instancias pertenecientes a la misma clase con índices indexes
		Devuelve matriz con una instancia de cada clase
		"""
		new_prot = np.zeros((self.n_class, self.n_col))
		# Sumo todos los valores atributo por atributo
		for i in indexes:
			i_class = self.classes.index(self.label[i])
			new_prot[i_class] += [self.values[i][j] for j in range(self.n_col)]

		# Divido entre el nº de prototipos
		for i in range(self.n_class):
			for j in range(self.n_col):
				new_prot[i][j] = new_prot[i][j]/self.n_prot[i]

		return new_prot



	def normalize(self):
		"""
		Método que normaliza el conjunto de datos
		"""
		min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
		self.values = min_max_scaler.fit_transform(self.values)

	def setClassesPairs(self):
		"""
		Devuelve todos los pares formados por las clases (one vs one)
		"""
		self.classes_pairs = itertools.combinations(self.classes, 2)




















