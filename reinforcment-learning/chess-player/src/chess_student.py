import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from my_code import *

def main(pars, alg='Q-learning', init='henormal', L2=0, seed=3):
    
    np.random.seed(seed)
    
    size_board = pars['size_board']
    
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """

    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left,
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left,
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1):
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)
    
    # Initialise the Weights storing them in dictionary W
    W = initW(x, N_a, init)
    
    # Network Parameters
    epsilon_0 = pars['epsilon_0']   #epsilon for the e-greedy policy
    beta = pars['beta']             #epsilon discount factor
    gamma = pars['gamma']           #SARSA Learning discount factor
    eta = pars['eta']               #learning rate
    N_episodes = pars['N_episodes'] #Number of games, each game ends when we have a checkmate or a draw
    window = pars['window']         #Window for Exponential Moving Average

    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction,
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])

    R_save = np.zeros(N_episodes)
    N_moves_save = np.zeros(N_episodes)
    W_count = np.zeros(N_episodes)

    # END OF SUGGESTIONS

    for n in range(N_episodes):
        epsilon_f = epsilon_0 / (1 + beta * n) # epsilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements
        W['dW'] = 0

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

        # Actions & allowed_actions
        a = np.concatenate([np.array(a_q1), np.array(a_k1)])
        allowed_a = np.where(a > 0)[0]

        while checkmate == 0 and draw == 0:

            R = 0  # Reward

            # Player 1

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)
            
            # Calculate Q values using neural network.
            Q, network = Q_values(x, W)
            
            # Select action based on epsilon greedy policy 
            a_agent = eGreedy(epsilon_f, allowed_a, Q)

            # Player 1 makes the action
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1  # Reward for checkmate
                
                # TD error with future reward = 0
                error = R - Q[a_agent] 
                
                # Apply back propagation to change weights.
                W = backProp(x, error, network, a_agent, eta, W, L2)

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1
                
                # TD error with future reward = 0
                error = R - Q[a_agent]
                
                # Apply back propagation to change weights.
                W = backProp(x, error, network, a_agent, eta, W, L2)

                if draw:
                    break

            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Update the parameters

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _ = Q_values(x_next, W)
            
            # Calculate the allowed actions for the next move.
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]
            
            if alg == 'Q-learning':
                # Select the next action according to greedy policy.
                a_agent_next = allowed_a[np.argmax(Q_next[allowed_a])]
            elif alg == 'SARSA':
                # Select the next action according to epsilon greedy policy.
                a_agent_next = eGreedy(epsilon_f, allowed_a, Q_next)
            elif alg == 'SARSA_0':
                # Same as SARSA but only initial epsilon is used.
                a_agent_next = eGreedy(epsilon_0, allowed_a, Q_next)
            
            # Calculate TD error
            error = R + (gamma * Q_next[a_agent_next]) - Q[a_agent]
            
            # Apply back propagation to change weights.
            W = backProp(x, error, network, a_agent, eta, W, L2)
                   
            i+=1
        
        # Adjust the records using exponential moving average.
        R_save[n] = ((1 - window) * R_save[n-1]) + (window * R)
        N_moves_save[n] = ((1 - window) * N_moves_save[n-1]) + (window * i)
        W_count[n] = ((1 - window) * W_count[n-1]) + (window * W['dW'])
        
        # If weights start to explode stop the run.
        if(W['dW'] > 0.1):
            break
        
        if (not n%10000) and (n != 0):
            print('Completed : ' + str(n) + ' episodes')
        
    return (R_save, N_moves_save, W_count)


if __name__ == '__main__':
    
    pars = {}
    pars['size_board'] = 4
    pars['epsilon_0'] = 0.1
    pars['beta'] = 0.00001
    pars['gamma'] = 0.85
    pars['eta'] = 0.01
    N_episodes = pars['N_episodes'] = 100000
    pars['window'] = 1/10000
    
    print('Testing optimal configuration with Q-learning')

    R_save, N_moves_save, W_count = main(pars)
    
    plt.figure('Average Reward')
    plt.plot(np.arange(N_episodes), R_save)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Reward')
    plt.title('Average Reward')
    
    plt.figure('Average Moves')
    plt.plot(np.arange(N_episodes), N_moves_save)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Moves')
    plt.title('Average Moves')
    
    plt.figure('Average Weight Change')
    plt.plot(np.arange(N_episodes), W_count)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Change in Weight')
    plt.title('Average Change in Weight')
    
    plt.show()
    
    print('Final reward is: ' + str(R_save[-1]))
