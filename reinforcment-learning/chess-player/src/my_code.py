import numpy as np

def Q_values(x, W):

    # Neural activation: input layer -> hidden layer
    act1 = np.dot(W['W1'], x) + W['bias_W1']
    out1 = act1 * (act1 > 0)

    # Neural activation: hidden layer -> output layer
    act2 = np.dot(W['W2'], out1) + W['bias_W2']
    out2 = act2 * (act2 > 0)
    
    Q = out2
    
    # YOUR CODE ENDS HERE
    return Q, (act1, out1, act2)


def initW(x, N_a, init):
    
    n_input_layer = x.shape[0]  # Number of neurons of the input layer.
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = N_a  # Number of neurons of the output layer. 

    # Initialise the weights between all layers.
    if init == 'uniform':  
        W1 = np.random.uniform(0,1, (n_hidden_layer, n_input_layer))
        W1 /= (n_input_layer * n_hidden_layer)     
        W2 = np.random.uniform(0,1, (n_output_layer, n_hidden_layer))
        W2 /= (n_hidden_layer * n_output_layer)
    elif init == 'henormal':
        W1 = np.random.randn(n_hidden_layer, n_input_layer) 
        W1 *= np.sqrt(2/(n_input_layer+n_hidden_layer))
        W2 = np.random.randn(n_output_layer, n_hidden_layer) 
        W2 *= np.sqrt(2/(n_hidden_layer+n_output_layer))
    elif init == 'xavier':
        W1 = np.random.randn(n_hidden_layer, n_input_layer) 
        W1 *= np.sqrt(1/(n_input_layer))
        W2 = np.random.randn(n_output_layer, n_hidden_layer) 
        W2 *= np.sqrt(1/(n_hidden_layer))
        
    # Initialise the biases for both layers
    bias_W1 = np.ones(n_hidden_layer) * 0.1
    bias_W2 = np.ones(n_output_layer) * 0.1
    
    W = {}
    W['W1'] = W1
    W['W2'] = W2
    W['bias_W1'] = bias_W1
    W['bias_W2'] = bias_W2
    W['dW'] = 0
    
    # YOUR CODES ENDS HERE
    
    return W


def eGreedy(epsilon, allowed_a, Q):
    
    eGreedy = int(np.random.rand() < epsilon)
    if eGreedy:
        a_agent = np.random.randint(0, allowed_a.shape[0])
        a_agent = allowed_a[a_agent]
    else:
        a_agent = allowed_a[np.argmax(Q[allowed_a])]
           
    #THE CODE ENDS HERE. 
    
    return a_agent


def backProp(x, error, network, a_agent, eta, W, L2):
    
    act1 = 0
    out1 = 1
    act2 = 2
    
    # Propogate the error from output to hidden layer
    out2delta = error * np.heaviside(network[act2][a_agent], 0)
    dW2 = eta * (out2delta * network[out1])   
    # Adjust the weights and biases
    W['W2'][a_agent] += dW2
    W['bias_W2'] += (eta * out2delta)
               
    # Propogate the error from hidden to input layer
    out1delta = (W['W2'][a_agent] * out2delta) * np.heaviside(network[act1], 0)
    dW1 = eta * np.outer(out1delta, x)
    # Adjust the weights and biases
    W['W1'] += dW1
    W['bias_W1'] += (eta * out1delta)
    
    W['dW'] += (np.mean(np.abs(dW2)) + np.mean(np.abs(dW1)))
    
    if L2:
        for key in W.keys():
           W[key] += (-L2 * W[key])
    
    return W