# Machine Learning — Estudos

Repositório para aprendizado progressivo de Machine Learning com Python e scikit-learn.

## Ambiente Virtual

Este projeto usa `.venv` para isolar as dependências Python do sistema.

### Criando o ambiente (primeira vez)

```bash
# Criar o ambiente virtual
python3 -m venv .venv
```

### Ativando o ambiente

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (CMD)
.venv\Scripts\activate.bat
```

Quando ativo, o terminal mostra o prefixo `(.venv)`:
```
(.venv) $
```

### Instalando as dependências

```bash
# Com o ambiente ativo
pip install -r requirements.txt
```

### Desativando o ambiente

```bash
deactivate
```

### Por que usar `.venv`?

| Sem venv | Com venv |
|----------|----------|
| Pacotes instalados globalmente | Pacotes isolados por projeto |
| Conflitos de versão entre projetos | Cada projeto tem suas versões |
| Difícil reproduzir o ambiente | `requirements.txt` garante reprodutibilidade |

---

## Módulos

| Módulo | Conteúdo | Status |
|--------|----------|--------|
| [Regressão Linear](./regressao_linear/) | Teoria, fórmulas e notebook prático com Iris dataset | ✅ |
| [Classificação](./classificacao/) | Teoria + comparação de Reg. Logística, KNN, Árvore de Decisão e Naive Bayes | ✅ |
| [Agrupamento](./agrupamento/) | Teoria + K-Means, DBSCAN e Hierárquico com métricas internas | ✅ |

## Estrutura

```
machinelearning-aprendizado/
├── .venv/                               # Ambiente virtual (não versionado)
├── .gitignore                           # Ignora .venv, __pycache__, etc.
├── requirements.txt                     # Dependências do projeto
├── README.md                            # Este arquivo
├── regressao_linear/
│   ├── README.md                        # Teoria e fórmulas
│   └── linear_regression_iris.ipynb     # Notebook prático
├── classificacao/
│   ├── README.md                        # Teoria, métricas e algoritmos
│   └── classificacao_iris.ipynb         # Notebook: Reg. Logística, KNN, Árvore, Naive Bayes
└── agrupamento/
    ├── README.md                        # Teoria, métricas internas e algoritmos
    └── agrupamento_iris.ipynb           # Notebook: K-Means, DBSCAN, Hierárquico
```

## Abrindo os notebooks

```bash
# 1. Ativar o ambiente
source .venv/bin/activate

# 2. Iniciar o Jupyter
jupyter notebook

# Ou abrir diretamente
jupyter notebook regressao_linear/linear_regression_iris.ipynb
```
