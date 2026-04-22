# Regressão Linear

## O que é?

Regressão Linear é um dos algoritmos mais fundamentais de Machine Learning supervisionado. Ele modela a **relação linear** entre uma ou mais variáveis independentes (features) e uma variável dependente contínua (target).

---

## Tipos

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Simples** | 1 feature → 1 target | Comprimento da pétala → Largura da pétala |
| **Múltipla** | N features → 1 target | Área + quartos + localização → Preço da casa |

---

## A Fórmula

### Regressão Linear Simples

```
ŷ = β₀ + β₁ · x
```

| Símbolo | Nome | Significado |
|---------|------|-------------|
| `ŷ` | Predição | Valor estimado pelo modelo |
| `β₀` | Intercepto (bias) | Valor de ŷ quando x = 0 (onde a reta cruza o eixo Y) |
| `β₁` | Coeficiente (slope) | Quanto ŷ muda para cada unidade de x |
| `x` | Feature | Variável de entrada |

### Regressão Linear Múltipla

```
ŷ = β₀ + β₁·x₁ + β₂·x₂ + ... + βₙ·xₙ
```

---

## Como o Modelo Aprende? (Mínimos Quadrados)

O objetivo é encontrar os valores de β₀ e β₁ que **minimizam o erro total** entre os valores reais (y) e os preditos (ŷ).

### Função de Custo — Erro Quadrático Médio (MSE)

```
MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
```

O scikit-learn resolve isso analiticamente (Ordinary Least Squares — OLS):

```
β = (XᵀX)⁻¹ · Xᵀy
```

---

## Métricas de Avaliação

### R² (Coeficiente de Determinação)
```
R² = 1 - (SS_res / SS_tot)
```
- **R² = 1.0** → modelo perfeito
- **R² = 0.0** → modelo não explica nada
- **R² < 0** → modelo pior que a média

### MSE (Mean Squared Error)
```
MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
```
Penaliza erros grandes. Unidade: target².

### RMSE (Root Mean Squared Error)
```
RMSE = √MSE
```
Mesma unidade do target — mais fácil de interpretar.

### MAE (Mean Absolute Error)
```
MAE = (1/n) · Σ |yᵢ - ŷᵢ|
```
Mais robusto a outliers que o MSE.

---

## Passo a Passo Conceitual

```
1. Coletar dados (x, y)
         ↓
2. Visualizar a relação (scatter plot)
         ↓
3. Dividir em treino e teste (train_test_split)
         ↓
4. Instanciar o modelo (LinearRegression())
         ↓
5. Treinar o modelo (model.fit(X_train, y_train))
         ↓
6. Avaliar o modelo (R², MSE, RMSE)
         ↓
7. Fazer predições (model.predict(X_new))
         ↓
8. Visualizar a reta de regressão
```

---

## Suposições da Regressão Linear

Para que o modelo funcione bem, idealmente:

1. **Linearidade** — relação entre x e y é linear
2. **Independência** — observações independentes entre si
3. **Homocedasticidade** — variância do erro é constante
4. **Normalidade dos resíduos** — erros distribuídos normalmente

---

## Arquivos deste módulo

| Arquivo | Conteúdo |
|---------|----------|
| `README.md` | Esta documentação teórica |
| `linear_regression_iris.ipynb` | Notebook completo com iris dataset |

---

## Referências

- [scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [scikit-learn: User Guide - Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
