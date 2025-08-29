import math

import numpy as np

def positional_encoding(max_len, d_model):
    """
    Gera matriz de Positional Encoding [max_len, d_model]
    """
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)   # (max_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # dimensões pares
    pe[:, 1::2] = np.cos(position * div_term)  # dimensões ímpares
    return pe

# Exemplo
pe = positional_encoding(max_len=10, d_model=16)
print(pe.shape)  # (10, 16)
print(pe[0])     # posição 0
print(pe[1])     # posição 1

def transforma_query(query, wq):
    """
    Transforma a consulta (query) para o espaço de atenção.
    """
    return np.dot(query, wq)

def transforma_key(key, wk):
    """
    Transforma a chave (key) para o espaço de atenção.
    """
    return np.dot(key, wk)

def transforma_value(value, wv):
    """
    Transforma o valor (value) para o espaço de atenção.
    """
    return np.dot(value, wv)


def softmax(x, axis=-1):
    """
    Implementação estável do softmax em NumPy.
    """
    x = x - np.max(x, axis=axis, keepdims=True)  # estabilidade numérica
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cria_mask(seq_len):
    """
    Cria uma máscara triangular inferior para a atenção própria.
    """
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices(seq_len, 1)] = -1e9
    return mask

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Calcula a atenção de produto escalar escalada.
    q: (seq_len_q, d_k)
    k: (seq_len_k, d_k)
    v: (seq_len_k, d_v)
    mask: opcional (seq_len_q, seq_len_k)
    """
    d_k = q.shape[-1]
    scores = np.dot(q, k.T) / np.sqrt(d_k)  # escala correta
    if mask is not None:
        scores = scores + (mask * -1e9)  # aplica máscara
    attn = softmax(scores, axis=-1)       # softmax nas chaves
    return np.dot(attn, v), attn          # retorna valores + pesos de atenção


def concatena_atencao(*args, mask=None):
    """
    Concatena as saídas da atenção.
    """
    concat = np.concatenate(args, axis=-1)  # concatena na última dimensão
    return concat

def linear(matriz_concatenada, w):
    """
    Aplica uma transformação linear simples.

    matriz concatenada: num_tokens X d_model *  num_heads

    w: d_model * num_heads X d_model
    """
    return np.dot(matriz_concatenada, w)

def add_and_norm(matriz, residual, eps=1e-6):
    """
    Aplica soma residual e normalização por vetor (LayerNorm simplificado).
    
    matriz: num_tokens x d_model
    residual: num_tokens x d_model
    """
    add = matriz + residual  # soma residual
    
    # normaliza por token (linha da matriz)
    mean = np.mean(add, axis=-1, keepdims=True)
    std = np.std(add, axis=-1, keepdims=True)
    norm = (add - mean) / (std + eps)
    
    return norm

def feed_forward_network(matriz, w1, b1, w2, b2):
    """
    Aplica uma rede feed-forward simples.
    matriz: num_tokens x d_model
    w1: d_model x d_ff
    b1: d_ff
    w2: d_ff x d_model
    b2: d_model
    """
    hidden = np.dot(matriz, w1) + b1
    hidden = np.maximum(0, hidden)  # ReLU
    output = np.dot(hidden, w2) + b2
    return output

import numpy as np

def cria_mask(num_linhas, num_colunas):
    """
    Cria uma máscara com metade inferior (diagonal) = 0
    e metade superior (diagonal) = 1.
    """
    mask = np.zeros((num_linhas, num_colunas))

    for i in range(num_linhas):
        for j in range(num_colunas):
            if j > i:
                mask[i, j] = 1
    return mask

def projecao_linear(matriz, w):
    """
    Aplica uma projeção linear a uma matriz.
    matriz de pesos: (d_model x d_output)
    retorno: num_tokens X d_output
    """
    return np.dot(matriz, w)
