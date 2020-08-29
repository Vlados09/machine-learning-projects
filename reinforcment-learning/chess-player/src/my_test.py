import numpy as np
import matplotlib.pyplot as plt
import decimal

from chess_student import main


def default_pars():

    pars = {}
    pars['size_board'] = 4
    pars['epsilon_0'] = 0.2
    pars['beta'] = 0.00005
    pars['gamma'] = 0.85
    pars['eta'] = 0.0035
    pars['N_episodes'] = 100000
    pars['window'] = 1/10000

    return pars

def optimal_pars():
    
    pars = {}
    pars['size_board'] = 4
    pars['epsilon_0'] = 0.2
    pars['beta'] = 0.0001
    pars['gamma'] = 0.85
    pars['eta'] = 0.01
    pars['N_episodes'] = 100000
    pars['window'] = 1/10000
    
    return pars


def test_param(pars, values, name, seed=3):

    N_episodes = pars['N_episodes']

    size = len(values)

    R_save = np.zeros((size, N_episodes))
    N_moves_save = np.zeros((size, N_episodes))
    W_count = np.zeros((size, N_episodes))

    str_values = [None]*size
    for i in range(size):
        str_values[i] = float_to_str(values[i])

    for i in range(size):

        print('Testing ' + name + ' = ' + str_values[i])

        pars[name] = values[i]

        R_save[i], N_moves_save[i], W_count[i] = main(pars, seed=seed)

    return R_save, N_moves_save, W_count


def test_param_n(pars, values, name, n):
    
    str_values = [None]*len(values)
    for i in range(len(values)):
        str_values[i] = float_to_str(values[i])

    seed = np.random.randint(1000, size=n)
    print('Testing itteration 0')
    R, N, W = test_param(pars, values, name)
    for s in range(len(seed)):
        print('Testing itteration ' + str(s))
        R_save, N_moves_save, W_count = test_param(pars, values, name, seed[s])
        for i in range(len(values)):
            R[i] = (R[i] + R_save[i])
            N[i] = (N[i] + N_moves_save[i])
            W[i] = (W[i] + W_count[i])

    R /= (n+1)
    N /= (n+1)
    W /= (n+1)
    
    show_results(R, N, W, str_values, name)

    return R, N, W


def test_SARSA_v_Q(pars, values, seed=3):

    N_episodes = pars['N_episodes']
     
    size = len(values)

    L2 = [0.00001]*3

    R_save = np.zeros((size, N_episodes))
    N_moves_save = np.zeros((size, N_episodes))
    W_count = np.zeros((size, N_episodes))

    for i in range(size):

        print('Testing ' + values[i])

        R_save[i], N_moves_save[i], W_count[i] = main(pars, alg=values[i], L2=L2[i], seed=seed)

    return R_save, N_moves_save, W_count


def test_SARSA_v_Q_n(pars, n, SARSA_0=False):
    values = ['Q-learning', 'SARSA']
    if SARSA_0:
        values.append('SARSA_0')
    seed = np.random.randint(1000, size=n)
    print('Testing itteration 0')
    R, N, W = test_SARSA_v_Q(pars, values)
    for s in range(len(seed)):
        print('Testing itteration ' + str(s))
        R_save, N_moves_save, W_count = test_SARSA_v_Q(pars, values, seed=seed)
        for i in range(R_save.shape[0]):
            R[i] = (R[i] + R_save[i])
            N[i] = (N[i] + N_moves_save[i])
            W[i] = (W[i] + W_count[i])

    R /= (n+1)
    N /= (n+1)
    W /= (n+1)

    show_results(R, N, W, values, 'algorithm')

    return R, N, W


def test_init(pars):

    N_episodes = pars['N_episodes']

    values = ['uniform', 'henormal', 'xavier']
    size = len(values)

    R_save = np.zeros((size, N_episodes))
    N_moves_save = np.zeros((size, N_episodes))
    W_count = np.zeros((size, N_episodes))

    for i in range(size):

        print('Testing ' + values[i])

        R_save[i], N_moves_save[i], W_count[i] = main(pars, init=values[i])

    show_results(R_save, N_moves_save, W_count, values, 'initialisation')
    
    
def test_explode(pars):
    
    N_episodes = pars['N_episodes']

    names = ['none', 'L2']
    values = [0, 0.00001]

    size = len(values)

    R_save = np.zeros((size, N_episodes))
    N_moves_save = np.zeros((size, N_episodes))
    W_count = np.zeros((size, N_episodes))

    for i in range(size):

        print('Testing SARSA_0 with ' + names[i] + ' normalisation')

        R_save[i], N_moves_save[i], W_count[i] = main(pars, alg='SARSA_0', L2=values[i])

    show_results(R_save, N_moves_save, W_count, names, 'normalisation')


def show_results(R_save, N_moves_save, W_count, data, testing):

    plt.figure('Average Reward')
    for i in range(len(data)):
        label = testing + ' : ' + data[i]
        plt.plot(np.arange(R_save[i].shape[0]), R_save[i], label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Reward')
    plt.title('Average Reward')
    plt.legend()

    plt.figure('Average Moves')
    for i in range(len(data)):
        label = testing + ' : ' + data[i]
        plt.plot(np.arange(N_moves_save[i].shape[0]), N_moves_save[i], label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Moves')
    plt.title('Average Moves')
    plt.legend()

    plt.figure('Average Weight Change')
    for i in range(len(data)):
        label = testing + ' : ' + data[i]
        plt.plot(np.arange(W_count[i].shape[0]), W_count[i], label=label)
    plt.xlabel('Number of Episodes')
    plt.ylabel('Change in Weight')
    plt.title('Average Change in Weight')
    plt.legend()
    
    plt.show()


def show_zoomed_reward(R_save):
    
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
    fig, ax = plt.subplots()
    ax.plot(np.arange(R_save[0].shape[0]), R_save[0])
    ax.plot(np.arange(R_save[0].shape[0]), R_save[1])
    ax.plot(np.arange(R_save[0].shape[0]), R_save[2])
    axins = zoomed_inset_axes(ax, 7, loc=5)
    axins.plot(np.arange(R_save[0].shape[0]), R_save[0])
    axins.plot(np.arange(R_save[0].shape[0]), R_save[1])
    axins.plot(np.arange(R_save[0].shape[0]), R_save[2])
    x1, x2, y1, y2 = 90000, 100000, 0.96, 1.01 # specify the limits
    axins.set_xlim(x1, x2) # apply the x-limits
    axins.set_ylim(y1, y2) # apply the y-limits
    plt.xticks(visible=False)
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
    
    plt.show()


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """

    ctx = decimal.Context()
    ctx.prec = 5

    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


# test_init(default_pars())
test_explode(optimal_pars())
# test_param_n(optimal_pars(), [0.00001, 0.0001, 0.001], 'beta', 0)
# test_param_n(optimal_pars(), [0.85, 0.9, 0.95], 'gamma', 0)
# R_save, _, _ = test_SARSA_v_Q_n(optimal_pars(), 0, SARSA_0=True)
# show_zoomed_reward(R_save) # Has to be with SARSA_0
