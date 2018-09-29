#######################################################
# Autora: Elena Romero Contreras
# Código basado en http://delta.cs.cinvestav.mx/~ccoello/EMOO/MOPSO.py
# Clase partícula e implementación del algoritmo MOPSO
# y sus funciones auxiliares
#######################################################

import random
import math
import numpy as np
import data as d


class Particle:
	"""
	Clase que representa una partícula compuesta por prototipos
	pertenecientes a cada una de las clases del conjunto de datos
	"""
	def __init__(self, prototypes, n_fit):
		self.X = prototypes				# matriz posición de la partícula
		self.n_class = len(prototypes)	# nº de clases (prototipos)
		self.n_col = len(prototypes[0])	# nº de atributos
		self.V = np.random.rand(self.n_class, self.n_col)  # matriz velocidad de la partícula

		self.fit = np.zeros(n_fit)		# Vector de valores fitness

		self.bestX = self.X[:,:]  		# Mejor posición individual
		self.bestF = np.zeros(n_fit)	# Mejor vector fitness indivitual



	def evaluate(self, data, fitness, n_fit, subset_index):
		"""
		Evalua la función fitness
		"""
		self.fit = fitness(data = data, prototypes = self.X, n_fit = n_fit, parts=subset_index)




	def updateVelocity(self,w,c1,c2, leader):
		"""
		Actualiza velocidad de la partícula
		"""
		for i in range(self.n_class):
			for j in range(self.n_col):
				r1=random.random()
				r2=random.random()            
				vel_cognitive = c1*r1*(self.bestX[i][j]-self.X[i][j])
				vel_social = c2*r2*(leader[i][j] - self.X[i][j])

				self.V[i][j] = w*self.V[i][j] + vel_cognitive + vel_social



	def updatePosition(self):
		"""
		Actualiza la posición de la partícula según la nueva velocidad
		"""
		for i in range(self.n_class):
			for j in range(self.n_col):
				self.X[i][j] = self.X[i][j] + self.V[i][j]
				

	def checkLimits(self):
		"""
		Comprobamos que la partícula no se sale de los límites
		"""
		for i in range(self.n_class):
			for j in range(self.n_col):
				if self.X[i][j] > 1:
					self.X[i][j] = 1
				elif self.X[i][j] < 0:
					self.X[i][j] = 0





#----------------------------------------------------------------------
#
#					ALGORITMO MOPSO
#
#----------------------------------------------------------------------

def MOPSO(data=None, fitness=None, n_fit=3, n_rep=100, num_part=100, max_iter=200, c1=1, c2=2):
	"""
	Algoritmo PSO multiobjetivo
	"""
	n_div = 30

	#Obtiene indices de los subconjuntos
	subset_index = getSubsetIndex(data.n_row, n_fit)
	#Generamos swarm de partículas
	swarm = initializeSwarm(data, num_part, n_fit, fitness, subset_index)

	# Almacenamos en el repositorio las posiciones de partículas no dominadas
	repX, repF = nonDominatedVectors(swarm)

	# Genera hipercubos del espacio de búsqueda explorado
	grid = generateHypercubes(repF, n_div)
	index, subindex = gridIndex(repF, grid, n_div)


	i = 1 
	while i < max_iter:
		w = 0.9 - ((0.9 - 0.4)/max_iter)*i 
		# Para cada partícula
		for j in range(num_part):
			#Selecciona líder
			h = selectLeader(repF, index)
			#Actualiza velocidad y posición
			swarm[j].updateVelocity(w,c1,c2,repX[h,:,:])
			swarm[j].updatePosition()
			swarm[j].checkLimits()

		# Evalua cada partícula y comprueba si es la mejor
		for j in range(num_part):
			swarm[j].evaluate(data, fitness, n_fit, subset_index)
			# Si la posición actual de la partícula es mejor que (domina a) la mejor posición almacenada,
			# se actualiza la posición
			if (all(swarm[j].fit[:] <= swarm[j].bestF[:]) & any(swarm[j].fit[:] < swarm[j].bestF[:])):
				swarm[j].bestF[:] = swarm[j].fit[:]
				swarm[j].bestX[:,:] = swarm[j].X[:,:]
			elif (all(swarm[j].fit[:] >= swarm[j].bestF[:]) & any(swarm[j].fit[:] > swarm[j].bestF[:])):
				pass
			else:
			# Si ninguna domina a la otra, seleccionamos una aleatoria
				if (random.randint(0,1) == 0):
					swarm[j].bestF[:] = swarm[j].fit[:]
					swarm[j].bestX[:,:] = swarm[j].X[:,:]


		# Actualizamos repositorio y representación geográfica en los hipercubos
		new_repX, new_repF = nonDominatedVectors(swarm)
		new_index, new_subindex = gridIndex(new_repF, grid, n_div)

		repX, repF, index, subindex = archiveController(repX, repF, new_repX, new_repF, index, subindex, new_index, new_subindex)

		# Si la nueva partícula insertada se queda fuera de los limites del grid, este tiene que ser recalculado
		# y cada partícula ha de ser recolocada
		if (sum(sum((grid[:,-2] > repF).astype(int))) >= 1  & sum(sum((grid[:,1] < repF).astype(int))) >= 1):
			grid = generateHypercubes(repF, n_div)
			index, subindex = gridIndex(repF, grid, n_div)

		# Si el archivo ha alcanzado el tamaño máximo, el procedimiento adaptativo del grid es invocado
		if (len(repF[:,0]) > n_rep):
			repF, repX, index, subindex = removeParticles(repF, repX, n_rep, index, subindex)

		i+=1

	return repX, repF
	

#-----------------------------------------------------
#			Funciones usadas
#-----------------------------------------------------

def initializeSwarm(data, n_part, n_fit, fitness, subset_index):
	swarm = []
	for i in range(n_part):
		# Obtenemos y mezclamos prototipos de cada clase
		prototypes = data.merge(data.getPrototypesIndexes())
		# Creamos nueva partícula a partir de los protipos
		swarm.append(Particle(prototypes, n_fit))
		# Evaluamos cada partícula
		swarm[i].evaluate(data,fitness, n_fit, subset_index)
		swarm[i].bestF = swarm[i].fit[:]

	return swarm

def getSubsetIndex(n_row, n_fit):
	"""
	Obtiene los índices de los subconjuntos en los que se divide el conjunto de datos
	"""
	return np.array_split(range(n_row),n_fit)

def nonDominatedVectors(swarm):
	"""
	Almacena en repositorio las posiciones y fitness de  las partículas no dominadas
	"""
	swarm = np.array(swarm)
	n_part = len(swarm) 	
	n_dom = np.zeros((n_part,1))
	index = []					# índices de partículas no dominadas

	# Comprueba si están dominadas
	for i in range(n_part):
		n_dom[i][0] = 0
		for j in range(n_part):
			if j!=i:
				if (all(swarm[j].fit[:] <= swarm[i].fit[:]) & any(swarm[j].fit[:] < swarm[i].fit[:])):
					n_dom[i][0] = n_dom[i][0] +1

		if n_dom[i][0] == 0:
			index.append(i)

	# Almacena las no dominadas
	repX = [swarm[index[i]].X[:,:] for i in range(len(index))] # Lista de posiciones (matrices): shape(nrep, nclass, n_col)
	repF = [swarm[index[i]].fit[:] for i in range(len(index))] # Lista de vectores fitness


	# convierto listas en arrays
	repF = np.array(repF)
	repX = np.array(repX)

	it = 0
	n_rep = len(repX)
	while (it < n_rep):
		uIdx = sum((2 == (0 == (repX - repX[it,:,:])).sum(1).sum(1)).astype(int))
		if uIdx>1:
			repX = np.delete(repX, it, axis = 0)
			repF = np.delete(repF, it,axis=0)
			n_rep = len(repX)
			it=0
		else:
			it = it + 1

	return repX, repF


def generateHypercubes(F, n_div):
	"""
	Genera hipercubos para el espacio de búsqueda explorado y localiza las partículas
	usando estos hipercubos como un sistema de coordenadas donde cada coordenada de la partícula 
	está definida según los valores de sus funciones objetivo
	"""
	grid = np.zeros((0, n_div))
	for i in range(len(F[0,:])):
		grid = np.vstack((grid, np.append(np.append(-np.inf, np.linspace(F[np.argmin(F[:,i]),i], F[np.argmax(F[:,i]),i], num = n_div-2)), np.inf)))
	return grid



def gridIndex(F, grid, n_div):

	n_fit = len(F[0,:]) 
	n_part = len(F[:,0])

	subindex = []
	for i in range(n_part):
		subIdx = []
		for j in range(n_fit):
			subIdx.append(min(np.where(F[i,j] <= grid[j,:])[0]))
		subindex.append(subIdx)

	subindex = np.array(subindex)

	coordinates = []
	for i in range(n_part):
		coordinates.append(tuple([int(coord) for coord in  subindex[i,:]])) 
	coordinates = tuple(coordinates)

	dimension = []
	for i in range(n_fit):
		dimension.append(len(grid[i,:]))
	dimension = tuple(dimension)

	index = []
	for i in range(n_part):
		index.append(np.ravel_multi_index(coordinates[i], dims = dimension, order = 'C'))

	return index, subindex


def selectLeader(repF, index):
	"""
	Selecciona lider entre las partículas del repositorio
	"""
	x = 10
	uniqueIdx = np.zeros((len(index),2))
	for i in range(len(index)):
		nIdx = np.array(index == index[i])
		nIdx = nIdx.astype(np.int)
		nIdx = nIdx.sum(axis = 0)
		uniqueIdx[i,:] = np.array(np.hstack((nIdx, i)))
    
	fitness = np.zeros((len(repF[:,0]),1))
	for i in range(len(repF[:,0])):
		fitness[i] = x/uniqueIdx[i,0]
		
	rouletteWheel = np.cumsum(fitness, axis = 0) #suma acumulada de fitness
	finalIdx = np.where(((rouletteWheel.max() - rouletteWheel.min())*random.random()) <= rouletteWheel)[0]
	h = finalIdx[0]
    
	return h



def deleteH(repF, index):
	x = 10
	uniqueIdx = np.zeros((len(index),2))
	for i in range(len(index)):
		nIdx = np.array(index == index[i])
		nIdx = nIdx.astype(np.int)
		nIdx = nIdx.sum(axis = 0)
		uniqueIdx[i,:] = np.array(np.hstack((nIdx, i)))
    
	fitness = np.zeros((len(repF[:,0]),1))
	for i in range(len(repF[:,0])):
		fitness[i] = x*uniqueIdx[i,0]
        
	rouletteWheel = np.cumsum(fitness, axis = 0)
	finalIdx = np.where(((rouletteWheel.max() - rouletteWheel.min())*random.random()) <= rouletteWheel)[0]
	h = finalIdx[0]
    
	return h


def archiveController(X, F, arcX, arcF, idxRep, subIdx, newIdxRep, newSubIdx):

	X = np.concatenate((X, arcX), axis = 0)    
	F = np.concatenate((F, arcF), axis = 0)  
	idxR = np.concatenate((idxRep, newIdxRep), axis = 0) 
	subidxR = np.concatenate((subIdx, newSubIdx), axis = 0)  
	nF = len(F[:,0])
	nD = np.zeros((nF,1))
	index = []
    
	for i in range(nF):
		nD[i,0] = 0
		for j in range(nF):
			if j != i:
				if (all(F[j,:] <= F[i,:]) & any(F[j,:] < F[i,:])):
					nD[i,0] = nD[i,0] + 1
		if nD[i,0] == 0:
			index.append(i)
    
	repX = X[index,:]
	repF = F[index,:]
	indexRep = idxR[index]
	subindicesRep = subidxR[index,:]    

	# Compruebo si hay repetidos y los elimino
	it = 0
	n_rep = len(repX)
	while (it < n_rep):
		uIdx = sum((2 == (0 == (repX - repX[it,:,:])).sum(1).sum(1)).astype(int))
		#uIdx = ((repX - repX[it,:,:]).sum(1).sum(1)==0).sum()>=2
		if uIdx>1:
			repX = np.delete(repX, it, axis = 0)
			repF = np.delete(repF, it,axis=0)
			n_rep = len(repX)
			it=0
		else:
			it = it + 1
            
	return repX, repF, indexRep, subindicesRep



def removeParticles(repF, repX, nRep, index, subindices):
	h = np.zeros((nRep,1))
	for i in range(nRep):
		h[i,0] = deleteH(repF, index)
		repF = np.delete(repF, h[i,0], axis = 0)
		repX = np.delete(repX, h[i,0], axis = 0)
		index = np.delete(index, h[i,0], axis = 0)
		subindices = np.delete(subindices, h[i,0], axis = 0)
    
	return repF, repX, index, subindices


