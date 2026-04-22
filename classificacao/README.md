# Classificação

## O que é?

Classificação é um problema de **aprendizado supervisionado** onde o objetivo é prever a **categoria (classe)** de uma observação, em vez de um valor contínuo como na regressão.

---

## Regressão vs Classificação

| Aspecto | Regressão | Classificação |
|---------|-----------|---------------|
| Saída | Valor contínuo (ex: 2.4 cm) | Categoria discreta (ex: "setosa") |
| Exemplo | Prever preço de imóvel | Prever espécie de flor |
| Métricas | MSE, RMSE, R² | Accuracy, Precision, Recall, F1 |

---

## Tipos de Classificação

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Binária** | 2 classes possíveis | Spam / Não spam |
| **Multiclasse** | 3+ classes possíveis | Espécie da flor (3 tipos) |
| **Multilabel** | Múltiplos rótulos simultâneos | Tags de um artigo |

---

## Algoritmos Abordados

### 1. Regressão Logística

Apesar do nome, é um algoritmo de **classificação**. Usa a função sigmoide para transformar uma combinação linear das features em uma probabilidade entre 0 e 1.

**Função Sigmoide:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Onde:**
```
z = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

**Decisão:**
```
se σ(z) ≥ 0.5  →  Classe 1
se σ(z) < 0.5  →  Classe 0
```

**Custo (Log Loss / Cross-Entropy):**
```
L = -(1/n) · Σ [ yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ) ]
```

---

### 2. K-Nearest Neighbors (KNN)

Classifica uma amostra pela **maioria das k amostras mais próximas** no espaço de features. Sem fase de treino explícita — memoriza os dados.

**Distância Euclidiana:**
```
d(p, q) = √( Σ (pᵢ - qᵢ)² )
```

**Decisão:** classe mais frequente entre os k vizinhos mais próximos.

**Trade-off do k:**
- k pequeno → modelo complexo, overfitting
- k grande → modelo simples, underfitting

---

### 4. Naive Bayes (Gaussiano)

Algoritmo probabilístico baseado no **Teorema de Bayes**, com a suposição ("naive") de que as features são **independentes entre si** dado a classe.

**Teorema de Bayes:**
```
P(classe | X) = P(X | classe) · P(classe) / P(X)
```

| Termo | Nome | Significado |
|-------|------|-------------|
| `P(classe \| X)` | Posterior | Probabilidade da classe dado os dados observados |
| `P(X \| classe)` | Verossimilhança | Probabilidade dos dados dado a classe |
| `P(classe)` | Prior | Probabilidade a priori da classe (frequência no treino) |
| `P(X)` | Evidência | Constante de normalização (mesma para todas as classes) |

**Decisão:** escolhe a classe com maior posterior:
```
ŷ = argmax P(classe) · Π P(xᵢ | classe)
```

**Versão Gaussiana** — assume distribuição normal para cada feature por classe:
```
P(xᵢ | classe) = (1 / √(2π·σ²)) · exp( -(xᵢ - μ)² / (2σ²) )
```

Onde `μ` e `σ²` são a média e variância da feature para aquela classe, estimadas no treino.

**Vantagens:**
- Muito rápido para treinar e predizer
- Funciona bem com poucos dados
- Naturalmente gera probabilidades calibradas

**Limitação:** a suposição de independência raramente é verdadeira, mas o modelo tende a funcionar bem mesmo assim.

---

### 3. Árvore de Decisão (Decision Tree)

Divide o espaço de features recursivamente usando perguntas binárias. Cada nó interno é uma pergunta, cada folha é uma classe.

**Critério de divisão — Gini Impurity:**
```
Gini = 1 - Σ pᵢ²
```

**Critério de divisão — Entropia:**
```
Entropia = -Σ pᵢ · log₂(pᵢ)
```

Onde `pᵢ` é a proporção da classe `i` no nó.

---

## Métricas de Avaliação

### Matriz de Confusão

```
                  Predito
                Pos    Neg
Real  Pos  |  TP  |  FN  |
      Neg  |  FP  |  TN  |
```

| Sigla | Nome | Significado |
|-------|------|-------------|
| TP | True Positive | Acertou positivo |
| TN | True Negative | Acertou negativo |
| FP | False Positive | Disse positivo, era negativo |
| FN | False Negative | Disse negativo, era positivo |

### Fórmulas das Métricas

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)        → "dos que previ como positivo, quantos eram?"

Recall    = TP / (TP + FN)        → "dos que eram positivo, quantos encontrei?"

F1-Score  = 2 · (Precision · Recall) / (Precision + Recall)
```

### Quando usar cada métrica?

| Métrica | Quando priorizar |
|---------|-----------------|
| **Accuracy** | Classes balanceadas |
| **Precision** | Custo alto de falso positivo (ex: spam — não queremos perder e-mails bons) |
| **Recall** | Custo alto de falso negativo (ex: diagnóstico de câncer — não queremos perder casos reais) |
| **F1** | Equilíbrio entre Precision e Recall |

---

## Passo a Passo Conceitual

```
1. Coletar dados (features X, classes y)
         ↓
2. Análise exploratória (distribuição das classes, correlações)
         ↓
3. Pré-processamento (normalização, encoding)
         ↓
4. Dividir treino / teste (train_test_split)
         ↓
5. Treinar o modelo (model.fit(X_train, y_train))
         ↓
6. Avaliar (accuracy, matriz de confusão, classification_report)
         ↓
7. Comparar algoritmos
         ↓
8. Fazer predições (model.predict(X_new))
```

---

## Suposições e Cuidados

| Problema | Causa | Solução |
|----------|-------|---------|
| **Overfitting** | Modelo muito complexo | Regularização, menos features, mais dados |
| **Underfitting** | Modelo muito simples | Modelo mais complexo, mais features |
| **Classes desbalanceadas** | Uma classe domina | Resampling, class_weight='balanced' |
| **Features em escalas diferentes** | KNN é sensível a escala | StandardScaler, MinMaxScaler |

---

## Arquivos deste módulo

| Arquivo | Conteúdo |
|---------|----------|
| `README.md` | Esta documentação teórica |
| `classificacao_iris.ipynb` | Notebook com Regressão Logística, KNN, Árvore de Decisão e Naive Bayes |

---

## Referências

- [scikit-learn: Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn: LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [scikit-learn: DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [scikit-learn: GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)
