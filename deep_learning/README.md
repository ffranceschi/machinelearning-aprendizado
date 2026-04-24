# Deep Learning com PyTorch

## O que é Deep Learning?

Deep Learning é um subcampo de Machine Learning baseado em **redes neurais artificiais** com múltiplas camadas. Essas camadas aprendem representações hierárquicas dos dados — cada camada extrai features progressivamente mais abstratas.

```
Dados brutos → Camada 1 → Camada 2 → ... → Camada N → Predição
               (bordas)   (formas)          (conceitos)
```

---

## Neurônio Artificial (Perceptron)

A unidade básica de uma rede neural imita o neurônio biológico:

```
        x₁ ──w₁──┐
        x₂ ──w₂──┤
        x₃ ──w₃──┼──→ Σ(wᵢxᵢ + b) ──→ f(z) ──→ saída
        ...       │
        xₙ ──wₙ──┘
```

**Fórmula:**
```
z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b = Wᵀx + b

saída = f(z)
```

| Símbolo | Nome | Papel |
|---------|------|-------|
| `x` | Input | Dados de entrada |
| `W` | Pesos (weights) | Parâmetros aprendidos |
| `b` | Bias | Desloca a ativação |
| `f` | Função de ativação | Introduz não-linearidade |

---

## Funções de Ativação

Sem ativação, empilhar camadas lineares equivale a uma só camada linear. As funções de ativação introduzem **não-linearidade**, permitindo aprender padrões complexos.

### ReLU (Rectified Linear Unit)
```
f(z) = max(0, z)
```
- A mais usada em camadas ocultas
- Rápida de calcular, resolve o vanishing gradient parcialmente

### Sigmoid
```
f(z) = 1 / (1 + e^(-z))    →    saída ∈ (0, 1)
```
- Usada na saída para classificação binária
- Problema: gradiente quase zero em extremos (vanishing gradient)

### Softmax
```
f(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)    →    Σ saídas = 1
```
- Usada na saída para classificação multiclasse
- Converte logits em distribuição de probabilidade

### Tanh
```
f(z) = (e^z - e^(-z)) / (e^z + e^(-z))    →    saída ∈ (-1, 1)
```
- Versão centrada em zero do Sigmoid

---

## Arquitetura de uma Rede Neural

```
Input Layer      Hidden Layers         Output Layer
    │                │                     │
  [x₁]         [h₁] [h₄]               [ŷ₁]
  [x₂]  ──→   [h₂] [h₅]  ──→  ...  →  [ŷ₂]
  [x₃]         [h₃] [h₆]               [ŷ₃]
```

| Camada | Papel |
|--------|-------|
| **Input** | Recebe os dados brutos |
| **Hidden** | Extrai representações intermediárias |
| **Output** | Produz a predição final |

**Fully Connected (Dense):** cada neurônio conectado a todos da camada anterior.

---

## Funções de Perda (Loss Functions)

Medem o erro entre predição `ŷ` e valor real `y`.

### Cross-Entropy (Classificação Multiclasse)
```
L = -(1/n) · Σᵢ Σₖ yᵢₖ · log(ŷᵢₖ)
```
Usada com Softmax na saída. No PyTorch: `nn.CrossEntropyLoss()` já inclui Softmax internamente.

### Binary Cross-Entropy (Classificação Binária)
```
L = -(1/n) · Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
```
No PyTorch: `nn.BCELoss()` ou `nn.BCEWithLogitsLoss()`.

### MSE (Regressão)
```
L = (1/n) · Σ (yᵢ - ŷᵢ)²
```
No PyTorch: `nn.MSELoss()`.

---

## Backpropagation

O algoritmo que treina a rede — calcula o gradiente da loss em relação a cada peso usando a **regra da cadeia** do cálculo diferencial.

```
Forward pass:   X → ŷ → L(ŷ, y)        (calcula a perda)
Backward pass:  ∂L/∂W ← retropropaga   (calcula gradientes)
Update:         W ← W - α · ∂L/∂W      (atualiza pesos)
```

O PyTorch faz isso automaticamente via `loss.backward()` + `optimizer.step()`.

---

## Otimizadores

### SGD (Stochastic Gradient Descent)
```
W ← W - α · ∂L/∂W
```
Simples, mas pode oscilar. Com `momentum`:
```
v ← β·v + (1-β)·∂L/∂W
W ← W - α·v
```

### Adam (Adaptive Moment Estimation)
```
m ← β₁·m + (1-β₁)·∂L/∂W        (média dos gradientes)
v ← β₂·v + (1-β₂)·(∂L/∂W)²     (variância dos gradientes)
W ← W - α · m̂ / (√v̂ + ε)
```
O mais usado na prática — adapta a taxa de aprendizado por parâmetro.

---

## PyTorch — Conceitos Fundamentais

| Conceito | Descrição | Equivalente NumPy |
|----------|-----------|-------------------|
| `torch.Tensor` | Array N-dimensional com suporte a GPU | `np.ndarray` |
| `autograd` | Diferenciação automática | — |
| `nn.Module` | Classe base para definir modelos | — |
| `DataLoader` | Carrega dados em mini-batches | — |
| `optimizer` | Algoritmo de otimização | — |

### Loop de Treinamento Padrão

```python
for epoch in range(n_epochs):
    # 1. Forward pass
    y_pred = model(X_batch)

    # 2. Calcular perda
    loss = criterion(y_pred, y_batch)

    # 3. Zerar gradientes antigos
    optimizer.zero_grad()

    # 4. Backward pass (calcular gradientes)
    loss.backward()

    # 5. Atualizar pesos
    optimizer.step()
```

---

## Passo a Passo Conceitual

```
1. Definir a arquitetura (nn.Module)
         ↓
2. Escolher loss function e otimizador
         ↓
3. Preparar dados (DataLoader, normalização)
         ↓
4. Loop de treino (forward → loss → backward → update)
         ↓
5. Avaliar no conjunto de validação/teste
         ↓
6. Ajustar hiperparâmetros (lr, n_layers, n_neurons)
```

---

## Hiperparâmetros Importantes

| Hiperparâmetro | Efeito | Valor típico |
|---------------|--------|--------------|
| `learning_rate` | Tamanho do passo no gradiente | 1e-3 a 1e-4 |
| `batch_size` | Amostras por atualização | 32, 64, 128 |
| `n_epochs` | Quantas vezes percorre os dados | 50–200 |
| `n_layers` | Profundidade da rede | 2–5 |
| `n_neurons` | Largura de cada camada | 64–512 |
| `dropout` | Regularização (desliga neurônios) | 0.2–0.5 |

---

## Arquivos deste módulo

| Arquivo | Conteúdo |
|---------|----------|
| `README.md` | Esta documentação teórica |
| `deep_learning_pytorch.ipynb` | Notebook: tensores, autograd, rede neural no Iris |

---

## Referências

- [PyTorch: Documentação oficial](https://pytorch.org/docs/stable/index.html)
- [PyTorch: Tutorial rápido (60 min)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch: nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [PyTorch: Autograd](https://pytorch.org/docs/stable/autograd.html)
