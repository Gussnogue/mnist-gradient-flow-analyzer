import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

def plot_loss_curve(losses):
    """Cria gráfico de linha da perda ao longo das épocas."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(losses))), y=losses, mode='lines', name='Loss'))
    fig.update_layout(title='Evolução da Perda', xaxis_title='Época', yaxis_title='Loss')
    return fig

def plot_gradient_heatmap(gradient_stats, layers, epochs):
    """
    Cria um heatmap com a média dos gradientes por camada ao longo das épocas.
    gradient_stats: lista de dicionários, cada um com 'layer_grad_means' (lista com média por camada na época)
    """
    # Converte para matriz (épocas x camadas)
    data = []
    for epoch_stats in gradient_stats:
        data.append(epoch_stats['layer_grad_means'])
    df = pd.DataFrame(data, columns=layers)
    fig = px.imshow(df.T, title='Média dos Gradientes por Camada (Heatmap)',
                    labels=dict(x='Época', y='Camada', color='Média do Gradiente'),
                    aspect='auto')
    return fig

def plot_gradient_boxplot(gradient_stats, layers, epoch):
    """
    Cria um boxplot dos gradientes em uma época específica.
    """
    values = gradient_stats[epoch]['all_gradients']  # lista de listas com gradientes individuais
    fig = go.Figure()
    for i, layer in enumerate(layers):
        fig.add_trace(go.Box(y=values[i], name=layer))
    fig.update_layout(title=f'Distribuição dos Gradientes - Época {epoch}',
                      yaxis_title='Valor do Gradiente')
    return fig

