import numpy as np


def features(p_q1, p_k1, p_k2, dfK2, s, check):

    size_board = s.shape[0]

    # Degrees of freedom of the Enemy King
    K2dof = len(np.where(dfK2 == 1)[0])

    s_k1 = np.array(s == 1).astype(int).reshape(-1)
    s_q1 = np.array(s == 2).astype(int).reshape(-1)
    s_k2 = np.array(s == 3).astype(int).reshape(-1)

    x = np.concatenate([s_k1, s_q1, s_k2, [check], [K2dof]])

    return x