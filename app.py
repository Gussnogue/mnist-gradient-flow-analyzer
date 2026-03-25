import streamlit as st
import torch
import numpy as np
from trainer import train_model
from utils import plot_loss_curve, plot_gradient_heatmap, plot_gradient_boxplot
from ai_explainer import explain_gradient_flow

st.set_page_config(page_title="Analisador de Fluxo de Gradientes", layout="wide")
st.title("📉 Analisador de Fluxo de Gradientes com IA Local")
st.markdown("Treine uma rede neural profunda, visualize gradientes e obtenha análises do **Hermes 3**.")

# ========== SIDEBAR: PARÂMETROS ==========
with st.sidebar:
    st.header("⚙️ Configurações da Rede")
    activation = st.selectbox("Função de ativação", ["relu", "sigmoid", "tanh", "leaky_relu"])
    depth = st.slider("Profundidade (camadas ocultas)", 2, 20, 10)
    hidden_size = st.slider("Tamanho oculto por camada", 64, 512, 256, step=64)
    learning_rate = st.slider("Taxa de aprendizado (learning rate)", 0.001, 0.1, 0.01, step=0.001)
    num_epochs = st.slider("Número de épocas", 1, 30, 10)

    if st.button("🚀 Iniciar Treinamento"):
        st.session_state['run_training'] = True

# ========== ÁREA PRINCIPAL ==========
if 'run_training' not in st.session_state:
    st.session_state['run_training'] = False

if st.session_state['run_training']:
    with st.spinner("Treinando a rede e coletando gradientes..."):
        losses, gradient_stats, layer_names = train_model(
            activation=activation,
            depth=depth,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs
        )
    st.success("Treinamento concluído!")

    # Armazenar resultados na sessão
    st.session_state['losses'] = losses
    st.session_state['gradient_stats'] = gradient_stats
    st.session_state['layer_names'] = layer_names
    st.session_state['activation'] = activation
    st.session_state['depth'] = depth
    st.session_state['hidden_size'] = hidden_size
    st.session_state['learning_rate'] = learning_rate
    st.session_state['run_training'] = False

# Se já houver resultados, exibir
if 'losses' in st.session_state:
    losses = st.session_state['losses']
    gradient_stats = st.session_state['gradient_stats']
    layer_names = st.session_state['layer_names']
    activation = st.session_state['activation']
    depth = st.session_state['depth']
    hidden_size = st.session_state['hidden_size']
    learning_rate = st.session_state['learning_rate']

    # Abas para visualização
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Perda", "🔥 Heatmap de Gradientes", "📦 Boxplot por Época", "🤖 Análise IA"])

    with tab1:
        st.plotly_chart(plot_loss_curve(losses), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_gradient_heatmap(gradient_stats, layer_names, len(losses)), use_container_width=True)

    with tab3:
        if len(gradient_stats) > 0:
            epoch_choice = st.slider("Selecione a época para boxplot", 0, len(gradient_stats)-1, 0)
            st.plotly_chart(plot_gradient_boxplot(gradient_stats, layer_names, epoch_choice), use_container_width=True)

    with tab4:
        st.subheader("🧠 Análise do Hermes 3")
        if st.button("Gerar Análise"):
            with st.spinner("Consultando IA local..."):
                analysis = explain_gradient_flow(
                    activation, depth, hidden_size, learning_rate,
                    losses, gradient_stats
                )
            st.markdown(analysis)

else:
    st.info("Configure os parâmetros na barra lateral e clique em **Iniciar Treinamento**.")

    