# 📐 Matrizes & Algoritmo Neural do Zero

Este repositório contém anotações teóricas detalhadas sobre álgebra linear (foco em matrizes e suas propriedades) e a implementação prática de um **Algoritmo Neural do Zero**, utilizando operações matriciais para fundamentar a arquitetura e a matemática por trás de redes neurais.

---

## 📚 Conteúdo do Repositório

### 1. 📝 Anotações Teóricas (`Matrizes.py`)
Anotações e resumos sobre conceitos fundamentais de matrizes e álgebra linear aplicados à computação e ciência de dados:
* **Definições e Tipos de Matrizes:** Matriz identidade, transposta, diagonal, simétrica e inversa.
* **Operações Fundamentais:** Adição, subtração, multiplicação por escalar e produto matricial ($A \times B$).
* **Propriedades Importantes:** Determinantes, autovalores e autoveetores (*eigenvalues/eigenvectors*).
* **Aplicações em Aprendizado de Máquina:** Transformações lineares, representação de dados e representação vetorial de pesos e *biases*.

### 2. 🧠 Projeto: Algoritmo Neural do Zero (`Projeto_2`)
Implementação de uma rede neural artificial construída sem o uso de bibliotecas de alto nível de *deep learning* (como TensorFlow ou PyTorch), utilizando matrizes para todas as etapas do fluxo de dados.

#### 🛠️ O que foi implementado:
* **Propagação Direta (*Forward Propagation*):** Cálculo matricial das camadas $Z = W \cdot X + b$ e aplicação das funções de ativação.
* **Funções de Ativação:** Sigmoide, ReLU e Softmax (vetorizadas).
* **Otimização:** Gradiente Descendente (*Gradient Descent*) para atualização dos pesos.

