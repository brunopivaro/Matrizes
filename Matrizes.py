# Matrizes, Determinantes, Autovalores e Autovetores

import numpy as np

# Escalares
escalar = np.array(8)
print(escalar)
print(escalar.shape) # Não tem dimensões, por ser um escalar

# Vetores
vetor = np.array([1, 2, 3])
print(vetor)
print(vetor.shape) # Possui dimensão, 3 linhas

# Sliccing o array de vetores
print(vetor[0:1])

# Matrizes
matriz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) # A Matriz é uma lista de listas ([[x], [y], [z]]])
print(matriz)
print(matriz.shape) # Possui dimensão, 3x3

# Sliccing o array de matriz
print(matriz[1][2])

# Tensores são arrays com mais de 3 dimensões (3, 3, 2, 1)

# Alterando o shape 
# - Alterando um vetor para uma matriz, no nosso exemplo temos um vetor (3, 0), queremos transformar em uma matriz (1, 3)
vetor_para_matriz = vetor.reshape(1, 3)
print(vetor_para_matriz)
print(vetor_para_matriz.shape)

# - Alterando um tensor para um vetor, iremos achatar o tensor
tensor = np.array([
    [[1, 2], [3, 4]], \
    [[5, 6], [7, 8]]
])

tensor_para_vetor = tensor.flatten()
print(tensor_para_vetor)

# - Alterando um tensor para uma matriz
tensor_para_matriz = tensor.reshape(2, 4)
print(tensor_para_matriz)

# Operações Element-Wise com Vetores

vetor1 = vetor + 5 # Peguei cada elemento do vetor e somei mais 5
vetor1 = np.add(vetor, 5) # Mesma maneira que o anterior, porém usando método add
print(vetor1)

# Operações Element-Wise com Matrizes

matriz1 = np.array([[1, 2], [3, 4]])
matriz2 = np.array([[5, 6], [7, 8]])

print(matriz1 + matriz2)

# Produto Matricial 
print(np.matmul(matriz1, matriz2))
# Ou
print(matriz1 @ matriz2)
# Ou
print(np.dot(matriz1, matriz2))

# Operações Element-Wise apenas funcionam em estruturas com mesmas dimensões, se temos uma matriz 2x2, ela apenas pode fazer operações com outra
# matriz 2x2, caso as dimensionalidades sejam diferentes, precisamos fazer operações com matrizes (como np.dot, np.matmul)

# Broadcasting em Numpy é um recurso que permite que operações aritméticas sejam realizadas em arrays de diferentes formas e tamanhos
# O Broadcasting estende o menor array para conseguir realizar operações com o array maior

# Matrix Product
# - O número de colunas da matriz esquerda deve ser igual ao número de linhas da matriz direita


# Transposta da Matriz
matriz_original = np.array([[1, 2], [3, 4]])
print(matriz_original)

matriz_transposta = matriz_original.T
print(matriz_transposta)

# Calculando a determinante
determinante = np.linalg.det(matriz_original)
print(determinante)

# Calculando a matriz inversa
matriz_inversa = np.linalg.inv(matriz_original)
print(matriz_inversa)

# Autovalores e Autovetores
matriz_auto = np.array([[2, 2, 1], [1, 3, 1], [1, 2, 2]])

w, v = np.linalg.eig(matriz_auto)
print("Autovalor: ",  w)
print("Autovetor: ", v)