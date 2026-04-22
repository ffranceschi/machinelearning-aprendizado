# Agrupamento (Clustering)

## O que é?

Agrupamento é um problema de **aprendizado não supervisionado** — o modelo recebe dados **sem rótulos** e deve descobrir sozinho a estrutura natural dos dados, separando as observações em grupos (clusters) onde elementos similares ficam juntos.

---

## Supervisionado vs Não Supervisionado

| Aspecto | Supervisionado | Não Supervisionado |
|---------|---------------|-------------------|
| Dados de treino | Features + rótulos (X, y) | Apenas features (X) |
| Objetivo | Aprender a prever y | Descobrir estrutura em X |
| Exemplos | Classificação, Regressão | Clustering, PCA |
| Avaliação | Compara predição com y real | Métricas internas (silhouette) |

---

## Algoritmos Abordados

### 1. K-Means

Particiona os dados em **k clusters** minimizando a variância interna de cada cluster (inércia). Iterativo e determinístico dado um ponto inicial.

**Algoritmo:**
```
1. Inicializar k centroides aleatoriamente
2. Atribuir cada ponto ao centroide mais próximo
3. Recalcular centroides como média dos pontos do cluster
4. Repetir 2-3 até convergência (centroides não mudam)
```

**Função objetivo (Inércia):**
```
J = Σ Σ ||xᵢ - μₖ||²
   k  xᵢ∈Ck
```

Onde `μₖ` é o centroide do cluster `k` e `||·||²` é a distância euclidiana ao quadrado.

**Método do Cotovelo (Elbow Method):**  
Treina K-Means para k = 1, 2, ..., n e plota a inércia. O "cotovelo" — ponto onde a queda de inércia desacelera — indica o k ideal.

**Limitações:**
- Exige definir k manualmente
- Sensível à inicialização (usar `init='k-means++'`)
- Assume clusters esféricos e de tamanho similar
- Sensível a outliers

---

### 2. DBSCAN

**Density-Based Spatial Clustering of Applications with Noise** — encontra clusters com base na **densidade** de pontos. Não exige definir k e detecta outliers automaticamente.

**Conceitos:**
```
ε (epsilon)   → raio de vizinhança de um ponto
min_samples   → mínimo de pontos dentro de ε para ser "core point"
```

**Tipos de ponto:**
| Tipo | Definição |
|------|-----------|
| **Core point** | Tem ≥ `min_samples` vizinhos dentro de ε |
| **Border point** | Dentro de ε de um core point, mas com < `min_samples` vizinhos |
| **Noise (outlier)** | Não é core nem border — classificado como -1 |

**Algoritmo:**
```
1. Para cada ponto, contar vizinhos dentro do raio ε
2. Marcar como core os que têm ≥ min_samples vizinhos
3. Conectar core points que estão dentro de ε entre si (mesmo cluster)
4. Border points entram no cluster do core mais próximo
5. Pontos restantes = ruído (-1)
```

**Vantagens:**
- Detecta clusters de formato arbitrário
- Não precisa definir k
- Identifica outliers nativamente

**Limitações:**
- Sensível a ε e min_samples (requer tuning)
- Dificuldade com clusters de densidades muito diferentes

---

### 3. Agrupamento Hierárquico (Hierarchical Clustering)

Constrói uma **hierarquia de clusters** sem precisar definir k antecipadamente. Pode ser visualizado como um **dendrograma**.

**Variante Aglomerativa (bottom-up):**
```
1. Cada ponto começa como seu próprio cluster
2. Mesclar os dois clusters mais próximos
3. Repetir até restar apenas 1 cluster
4. Cortar o dendrograma na altura desejada para obter k clusters
```

**Critérios de ligação (linkage):**
| Critério | Distância entre clusters A e B |
|----------|-------------------------------|
| **single** | min( d(a, b) ) para a∈A, b∈B |
| **complete** | max( d(a, b) ) para a∈A, b∈B |
| **average** | média de todas as distâncias entre A e B |
| **ward** | minimiza a variância total dentro dos clusters |

**Ward** é o mais usado — tende a produzir clusters balanceados.

---

## Métricas de Avaliação

Como não há rótulos verdadeiros, usamos **métricas internas**:

### Silhouette Score

Para cada ponto `i`:
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

| Símbolo | Significado |
|---------|-------------|
| `a(i)` | Distância média de `i` aos outros pontos do **mesmo** cluster (coesão) |
| `b(i)` | Distância média de `i` aos pontos do **cluster mais próximo** (separação) |

**Interpretação:**
- `s ≈ +1` → ponto bem alocado no cluster correto
- `s ≈ 0` → ponto na fronteira entre clusters
- `s ≈ -1` → ponto provavelmente no cluster errado

A **Silhouette Score média** (de todos os pontos) avalia a qualidade geral do agrupamento.

### Davies-Bouldin Index

```
DB = (1/k) · Σ max_{j≠i} ( (σᵢ + σⱼ) / d(cᵢ, cⱼ) )
```

Onde `σᵢ` é o espalhamento médio do cluster `i` e `d(cᵢ, cⱼ)` a distância entre centroides.  
**Menor é melhor** (clusters compactos e separados).

### Calinski-Harabasz Index

```
CH = (SS_between / (k-1)) / (SS_within / (n-k))
```

Razão entre dispersão entre clusters e dispersão dentro dos clusters.  
**Maior é melhor**.

---

## Passo a Passo Conceitual

```
1. Coletar dados (apenas features X, sem rótulos)
         ↓
2. Análise exploratória (distribuições, correlações)
         ↓
3. Pré-processamento (normalização obrigatória para K-Means e DBSCAN)
         ↓
4. Escolher número de clusters (Elbow Method, Silhouette)
         ↓
5. Treinar o modelo (model.fit(X))
         ↓
6. Avaliar com métricas internas (Silhouette, Davies-Bouldin)
         ↓
7. Visualizar clusters (scatter plot, dendrograma)
         ↓
8. Comparar algoritmos e interpretar os grupos encontrados
```

---

## Suposições e Cuidados

| Problema | Causa | Solução |
|----------|-------|---------|
| **Escala diferente entre features** | K-Means usa distância euclidiana | Sempre normalizar (StandardScaler) |
| **k errado no K-Means** | Elbow não é sempre claro | Complementar com Silhouette Score |
| **DBSCAN com ε ruim** | Clusters muito grandes ou tudo virou ruído | Usar k-distance plot para calibrar ε |
| **Clusters não esféricos** | K-Means falha | Usar DBSCAN ou Hierárquico |
| **Muitas dimensões** | Distâncias perdem significado** | Reduzir dimensionalidade (PCA) antes |

---

## Arquivos deste módulo

| Arquivo | Conteúdo |
|---------|----------|
| `README.md` | Esta documentação teórica |
| `agrupamento_iris.ipynb` | Notebook com K-Means, DBSCAN e Hierárquico |

---

## Referências

- [scikit-learn: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [scikit-learn: KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn: DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [scikit-learn: AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [scikit-learn: silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
