import numpy as np

def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_

def multiplicative_attention_score(decoder_hidden_state, encoder_hidden_states, W_mult):
    tmp = decoder_hidden_state.T @ W_mult
    attention_scores = tmp @ encoder_hidden_states
    return attention_scores

def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    scores = multiplicative_attention_score(decoder_hidden_state, encoder_hidden_states, W_mult)
    attention_weights = softmax(scores)
    context_vector = encoder_hidden_states @ attention_weights.T
    return context_vector

def additive_attention_score(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    th = np.tanh(W_add_enc @ encoder_hidden_states + W_add_dec @ decoder_hidden_state)
    scores = np.dot(v_add.T, th)
    return scores

def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    scores = additive_attention_score(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec)
    attention_weights = softmax(scores)
    context_vector = encoder_hidden_states.dot(attention_weights.T)
    return context_vector

if __name__ == "__main__":
    # Задаём фиксированные входные данные:
    # decoder_hidden_state: (3, 1)
    decoder_hidden_state = np.array([[1],
                                     [2],
                                     [3]])

    # encoder_hidden_states: (3, 4) – каждая колонка — скрытое состояние энкодера
    encoder_hidden_states = np.array([[1, 2, 3, 4],
                                      [5, 6, 7, 8],
                                      [9, 10, 11, 12]])

    # W_mult: (3, 3). Для простоты возьмём единичную матрицу (identity), чтобы не менять decoder_hidden_state.
    W_mult = np.eye(3)

    print("decoder_hidden_state.shape:", decoder_hidden_state.shape)       # (3, 1)
    print("encoder_hidden_states.shape:", encoder_hidden_states.shape)     # (3, 4)
    print("W_mult.shape:", W_mult.shape)                                   # (3, 3)

    # Вычисляем оценки внимания:
    scores = multiplicative_attention_score(decoder_hidden_state, encoder_hidden_states, W_mult)
    print("Attention scores (до softmax):", scores)
    print("Scores shape:", scores.shape)   # Ожидается (1, 4)

    # Применяем multiplicative_attention для получения контекстного вектора:
    context_vector = multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult)
    print("Context vector (итоговый attention vector):")
    print(context_vector)
    print("Context vector shape:", context_vector.shape)  # Ожидается (3, 1)

