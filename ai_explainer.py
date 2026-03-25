import requests

def explain_gradient_flow(activation, depth, hidden_size, learning_rate, losses, gradient_stats):
    """Envia estatísticas dos gradientes para o Hermes 3 e retorna análise."""
    chat_url = "http://localhost:1234/v1/chat/completions"  # Altere se necessário

    # Prepara dados resumidos
    final_loss = losses[-1] if losses else 0
    if len(gradient_stats) >= 2:
        first_epoch_grads = gradient_stats[0]['layer_grad_means']
        last_epoch_grads = gradient_stats[-1]['layer_grad_means']
    else:
        first_epoch_grads = last_epoch_grads = []

    prompt = f"""
    Você é um especialista em redes neurais profundas.
    Analise o comportamento dos gradientes durante o treinamento de uma rede com:
    - Função de ativação: {activation}
    - Profundidade: {depth} camadas
    - Tamanho oculto: {hidden_size}
    - Taxa de aprendizado: {learning_rate}

    Perda final: {final_loss:.4f}

    Médias dos gradientes (em valor absoluto) na primeira época: {first_epoch_grads}
    Médias dos gradientes na última época: {last_epoch_grads}

    Com base nessas informações:
    - Identifique se há indícios de vanishing ou exploding gradients.
    - Explique como a função de ativação escolhida influencia esse comportamento.
    - Sugira alterações (taxa de aprendizado, inicialização, normalização) para melhorar o treinamento.
    - Escreva em português, de forma didática.
    """
    payload = {
        "model": "hermes-3-llama-3.2-3b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 800
    }
    try:
        response = requests.post(chat_url, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Erro ao obter explicação: {response.text}"
    except Exception as e:
        return f"Falha na conexão com o LM Studio: {e}"

        