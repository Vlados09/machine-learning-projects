import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from pylab import pcolor, show, colorbar, xticks, yticks
import tensorflow_datasets as tfds

print("Extracting training data...")
emnist = tfds.load('emnist/letters')
emnist_tr = emnist['train']

# Convert tensorflow dataset to numpy array:
train, labels = [], []
for point in emnist_tr.as_numpy_iterator():
    train.append(point['image'].flatten())
    labels.append(point['label'])
train = np.array(train).T
labels = np.array(labels)

def init_params(a):

    params = {}

    params['eta'] = 0.9 # learning rate
    params['leaky_eta'] = 1/5000 # proportion of leaky learning
    params['eta_alpha'] = 0.5
    params['window'] = 1000
    params['letters'] = a # number of prototypes (outputs)
    params['itter'] = params['letters']*1000
    params['corr_threshold'] = 0.8

    return params


def normalise(arr):

    norm_arr = arr/LA.norm(arr)

    return norm_arr


def init_weights(params, data):

    letters = params['letters']

    [n,m] = np.shape(data)

    # Weight matrix (rows = output neurons, cols = input neurons):
    W = np.random.rand(letters,n)

    W = normalise(W)

    return W

def competetive(W, data, params, leaky=False, noise=False, dynamic_eta=False, conscience=False):

    eta0 = eta = params['eta']
    leaky_eta = params['leaky_eta']
    eta_alpha = params['eta_alpha']
    window = params['window']
    itter = params['itter']
    letters = params['letters']

    [n,m] = np.shape(data)  # number of pixels and number of training data

    counter = np.zeros(letters) # counter for the winner neurons
    wCount = np.ones(itter+1) * 0.25 # running avg of the weight change over
    prob = np.zeros(letters)

    print("Learning...")
    for t in range(1,itter):

        if dynamic_eta: # if learning rate is set to decay
            eta = eta0*(t**(-eta_alpha))

        # get randomly generated index in the input range:
        i = math.ceil(m*np.random.rand())-1

        x = data[:,i] # pick training instance using the random index

        h = W.dot(x)/letters # get ouput firing

        # scale values to be between 0 and 1:
        h = np.interp(h, (h.min(), h.max()), (0, 1))

        k = np.argmax(h) # index of the winning neuron

        if conscience: # if conscience algorithm is used
            out = np.zeros(letters)
            out[k] = 1
            prob += 0.001*(out-prob) # adjust the probabilities
            bias = 1*((1/letters)-prob) # calcualate the bias for each neuron
            h -= (1-bias) # subtract the bias from the output
            k = np.argmax(h) # select a new winner

        if noise: # if noise is used
            # longtail distribution
            nois = np.random.lognormal(0.01,0.65,letters)/100
            h += nois
            k = np.argmax(h)


        counter[k] += 1 # increment counter for winner neuron

        # calculate the change in weights for the k-th output neuron:
        dw = eta * (x.T - W[k,:])
        W[k,:] += dw # weights for k-th output are updated

        abs_dw = np.mean(np.abs(dw))
        wCount[t] = wCount[t-1] * (((window-1)/window)+abs_dw/window)

        if leaky: # if leaky learning is used
            # calculate the change in weights for all other neurons:
            dw = leaky_eta * (x.T - W[not k,:])
            W[not k,:] = W[not k, :] + dw # weights for all other are updated

    return W, wCount, counter


def correlation(W, data, params):

    letters = params['letters']

    corr = np.zeros((letters, letters))
    threshold = 0.5

    data = data.T

    for x in data:
        out = W.dot(x) # run the data point trough the network
        # scale values to be between 0 and 1:
        out = np.interp(out, (out.min(), out.max()), (0, 1))

        state = np.zeros(letters)
        state[out <= threshold] = -1
        state[out > threshold] = 1

        corr += np.outer(state, state)

    corr /= data.shape[0]

    return corr


def output_noise(W, params):

    letters = params['letters']

    out_noise = np.zeros(letters)

    dim = W[0, :].shape[0]
    h = w = int(math.sqrt(dim))

    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]

    for i in range(0,letters):
        image = W[i].reshape((h,w))
        noise = np.sum(np.sum(np.absolute(convolve2d(image, M))))
        noise *= math.sqrt(0.5 * math.pi) / (6 * (w-2) * (h-2))
        out_noise[i] = noise*1000000

    return out_noise


def display_results(W, wCount, params, counter, corr, out_noise):

    letters = params['letters']
    itter = params['itter']
    corr_threshold = params['corr_threshold']

    # Plot all prototypes
    n_plot = math.sqrt(letters)

    plt.figure(figsize=(15,15))

    for i in range(letters):
        plt.subplot(n_plot,n_plot,i+1)
        ax = plt.gca()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_title("Neuron " + str(i))

        y_dis = -0.3
        label = "Fired: " + str(int(counter[i]))+ " times" + "\n"

        distortion = (W[i]>np.mean(W[i])).sum()

        if out_noise[i] > 7:
            y_dis -= 0.1
            label += "Dead unit: " + str(round(out_noise[i],1)) + "\n"
        elif distortion > 2000:
            y_dis -= 0.1
            label += "Distorted: " + str(distortion)+ "\n"

        """
        corr_idx = ((corr[i]>corr_threshold) & (corr[i] != 1)).sum()
        if corr_idx.size:
            y_dis -= 0.1
            label += "Corr: " + np.array2string(corr_idx, separator=',') + "\n"
        """

        dim = W[0, :].shape[0]
        h = w = int(math.sqrt(dim))

        ax.text(0.5, y_dis,label,size=10, ha="center", transform=ax.transAxes)
        plt.imshow(W[i,:].reshape((w, h), order = 'F'),interpolation = 'nearest', cmap='inferno')
        plt.subplots_adjust(bottom=0.03, right=0.8, top=0.9)

    plt.show()


    # Display the correlation matrix
    pcolor(corr)
    colorbar()
    if(letters <= 16):
        yticks(np.arange(0.5,letters+0.5),range(0,letters))
        xticks(np.arange(0.5,letters+0.5),range(0,letters))
    show()

    # Adjust the correaltion matrix to show only higly correlated neurons
    corr[corr>corr_threshold] = 1
    corr[corr<=corr_threshold] = 0
    # Display the new correlation matrix
    pcolor(corr)
    colorbar()
    if(letters <= 16):
        yticks(np.arange(0.5,letters+0.5),range(0,letters))
        xticks(np.arange(0.5,letters+0.5),range(0,letters))
    show()

    plt.figure(figsize=(15,8))

    # Plot running average
    plt.subplot(2,2,1)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.plot(wCount[0:itter], linewidth=2.0, label='rate')

    # Plot firing rates of neurons
    plt.subplot(2,2,2)
    y_pos = np.arange(0,len(counter))
    plt.bar(y_pos,counter,align='center',alpha=0.5)
    plt.show()


params = init_params(36) # initialise the parameters with a give number of output units.
data = normalise(train)  # normalise the data
W = init_weights(params, data) # initialise the weights
end_W, wCount, counter = competetive(np.copy(W), data, params, conscience=True,
                                     dynamic_eta=True, noise = False, leaky = False) # run the algorithm
corr = correlation(end_W, data, params) # calculate correlation
out_noise = output_noise(end_W, params) # calculate noisiness in the image
display_results(end_W, wCount, params, counter, corr, out_noise) # display results



# Following code was used for the testing puprouses.
"""
units = np.arange(4,10)**2
units = np.insert(units, 0, 10)

a_itter = len(units)
a_step = 500
b_itter = 9
rep_itter = 10

for a in range(0,a_itter):
    letters = units[a]
    print("Testig for " + str(letters) + " output units")
    ave_std = np.empty(a_itter)
    ave_corr = np.empty(a_itter)
    ave_dead = np.empty(a_itter)
    ave_diss = np.empty(a_itter)

    stdC = 0
    corrC = 0
    deadC = 0
    dissC = 0

    for rep in range(0, rep_itter):

        params = init_params(letters)
        data = normalise(train)
        W = init_weights(params, data)
        end_W, wCount, counter = competetive(np.copy(W), data, params, conscience=True, dynamic_eta=True)

        stdC += np.std(counter)

        corr = correlation(end_W, data, params)
        corr_out = corr[np.triu_indices(letters, k = 1)]
        corr_out = corr_out[corr_out > 0.8]
        corrC += int(len(corr_out)/2)

        out_noise = output_noise(end_W, params)
        dead = out_noise>6
        deadC += (dead).sum()

        for out in range(0,letters):
            distortion = (end_W[out]>np.mean(end_W[out])).sum()
            if (distortion > 2000) & (not dead[out]):
                dissC += 1


    ave_corr[a] = ((corrC/rep_itter)/letters)*100
    print("Average correlated units: " + str(ave_corr[a]))
    ave_dead[a] = ((deadC/rep_itter)/letters)*100
    print("Average dead units: " + str(ave_dead[a]))
    ave_diss[a] = ((dissC/rep_itter)/letters)*100
    print("Average unit distortion: " + str(ave_diss[a]))
"""
