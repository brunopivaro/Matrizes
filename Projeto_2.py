# Bibliotecas
import numpy as np

# Escrevendo o Algoritmo
class AlgoritmoNeuralNetwork:

    def __init__(self, taxa_aprendizado, numero_interacoes):

        self.taxa_aprendizado = taxa_aprendizado
        self.numero_interacoes = numero_interacoes

        self.pesos = None
        self.bias = None
    
    # Formato 0 ou 1 para ser interpretado pelo algoritmo
    def func_activation_sigmoid(self, pred):

        sig = 1 / (1 + np.exp(-pred))

        return sig
    
    # Construindo o método Fit
    def fit(self, X, y):

        # Extrai o shape dos valores de entrada
        num_registros, num_atributos = X.shape

        # Inicializa a matriz de pesos com valores zero
        self.pesos = np.zeros(num_atributos)

        # Inicializa o bias com valor zero
        self.bias = 0

        # Loop de treinamento
        for i in range(self.numero_interacoes):

            print("Interação ", i)

            # Primeira parte do método fit é o foward pass

            # Faz a previsão usando os valores de X (dados de entrada), pesos e bias (produto escalar de X e pesos + bias)
            previsao = np.dot(X, self.pesos) + self.bias
            print("Previsão antes da função de ativação: ", previsao)

            # Utilizando a função sigmoide 
            previsao_final = self.func_activation_sigmoid(previsao)
            print("Previsão após a função de ativação: ", previsao_final)

            # Calculando o erro
            erro = (previsao_final - y)
            print("O erro é ", erro)

            # Segunda parte do método fit é o backward pass
            
            # Gradientes da matriz de pesos e do bias
            dw = (1 / num_registros) * np.dot(X.T, erro)
            db = (1 / num_registros) * np.sum(previsao_final - y)

            # Atualizando pesos e bias
            self.pesos -= self.taxa_aprendizado * dw
            print("Valore dos pesos ", self.pesos)

            self.bias -= self.taxa_aprendizado * db
            print("Valore de bias ", self.bias)
    
    # Método para as previsões
    def predict(self, X):
        
        # Faz as previsões com os dados de entrada e os valores atualizados dos pesos e do bias
        previsao = np.dot(X, self.pesos) + self.bias

        # Aplica a função sigmóide
        previsao_final = self.func_activation_sigmoid(previsao)

        # Aplica cut-off e converte probabilidades para 0 ou 1
        classe_prevista = [1 if i > 0.5 else 0 for i in previsao_final]

        return classe_prevista



# Preparando dados de entrada (Treino e Teste)
dados_entrada = np.array([[1, 2.5], [2, 3], [3, 5], [1, 4], [1.5, 2], [5, 6], [6, 7], [4, 5.5]])

# Os dados de saída representam, por exemplo, se uma transação foi suspeita ou não, são dados históricos
dados_saida = np.array([0, 0, 1, 0, 1, 0, 1, 1])


# Definindo hiperparametros 
taixa_aprendizagem = 0.01
numero_interacoes = 1000

# Criando modelo a partir da classe
modelo = AlgoritmoNeuralNetwork(taixa_aprendizagem, numero_interacoes)

# Treinando o modelo
modelo.fit(dados_entrada, dados_saida)


# Avaliação do modelo
X_testes = np.array([[1.5, 2], [4, 5.5]])
y_testes = np.array([0, 1])

previsoes_teste = modelo.predict(X_testes)