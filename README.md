# 📉 Analisador de Fluxo de Gradientes com IA Local

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com)
[![LM Studio](https://img.shields.io/badge/LM_Studio-0A0A0A?style=flat-square)](https://lmstudio.ai)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

> **Ferramenta didática com PyTorch e IA local:** treina uma rede neural profunda no dataset MNIST, coleta estatísticas de gradientes (heatmap, boxplot) e utiliza o **Hermes 3** (via LM Studio) para diagnosticar vanishing/exploding gradients e sugerir melhorias. Interface interativa com Streamlit.

---

## 🛠️ Stack Principal

| **Linguagem** | **Bibliotecas de ML** | **IA Local** | **Visualização** |
|---------------|-----------------------|--------------|------------------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) | ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white) | ![LM Studio](https://img.shields.io/badge/LM_Studio-0A0A0A?style=flat-square) ![Hermes 3](https://img.shields.io/badge/Hermes_3-FFD700?style=flat-square) | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white) |

---

## 📌 Sobre o Projeto

Este projeto demonstra o comportamento dos gradientes em redes neurais profundas. O usuário define a arquitetura (função de ativação, profundidade, tamanho oculto) e treina uma rede no **MNIST**. Durante o treinamento, são coletadas estatísticas dos gradientes (médias por camada e distribuição). Ao final, gráficos interativos (Plotly) permitem visualizar:

- Evolução da perda (loss)
- Heatmap dos gradientes ao longo das épocas
- Boxplot dos gradientes por camada em uma época específica

Além disso, um botão aciona o modelo **Hermes 3** (rodando localmente no LM Studio) para analisar os dados e fornecer um diagnóstico em português, identificando indícios de *vanishing* ou *exploding gradients* e sugerindo ajustes de hiperparâmetros.

---

## 🧠 Como Funciona

1. **Configuração:** o usuário escolhe função de ativação (ReLU, sigmoid, tanh, Leaky ReLU), profundidade, tamanho oculto e taxa de aprendizado.
2. **Treinamento:** uma rede neural profunda é treinada com um batch do MNIST. A cada época, os gradientes são armazenados.
3. **Visualização:** gráficos de perda, heatmap de gradientes e boxplot por camada/época.
4. **Análise com IA:** os dados resumidos (perda final, médias dos gradientes na primeira e última época) são enviados ao Hermes 3, que retorna uma explicação didática e recomendações.

---

## 🚀 Como Executar

### Pré‑requisitos
- Python 3.9+
- **LM Studio** com modelo **Hermes 3** carregado e servidor ativo (porta 1234)
- (Opcional) GPU para treinamento mais rápido

### Passo a passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Gussnogue/gradient-flow-analyzer.git
   cd gradient-flow-analyzer

2. **Crie um ambiente virtual e ative‑o**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```
3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```
4. **Execute a interface**
   ```bash
   streamlit run app.py
   ```
# 📁 Estrutura do Projeto
   ```bash
   gradient-flow-analyzer/
│
├── app.py                 # Aplicação Streamlit (interface)
├── trainer.py             # Lógica de treinamento e coleta de gradientes
├── ai_explainer.py        # Integração com Hermes 3
├── utils.py               # Funções de plotagem (Plotly)
├── requirements.txt       # Dependências
├── README.md              # Este arquivo
└── .env                   # (opcional) Configuração do LM Studio
   ```
# 📄 Licença
MIT License – sinta‑se à vontade para usar, modificar e distribuir.

# Referências:
Dataset: MNIST – Yann LeCun, Corinna Cortes, Christopher J.C. Burges
Modelo Local: Hermes 3 – Nous Research (via LM Studio)
