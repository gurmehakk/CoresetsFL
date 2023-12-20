import math
import numpy as np
# import cupy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = pd.read_csv('/home/shubham/gurmehak/DDcoresets/train.csv', low_memory=False)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:10000].T
Y_dev = data_dev[0]
Y_dev = Y_dev.T
X_dev = data_dev[1:n]
X_dev = X_dev / 255
X_dev = X_dev.T

data_train = data[10000:].T
Y_train = data_train[0]
Y_train = Y_train.T
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape
X_train = X_train.T
print(X_train.shape)
print(X_dev.shape)
epsilon = 0.5
delta = 0.5

def init_params():
    W1 = np.random.rand(500, 784) - 0.5
    b1 = np.random.rand(500, 1) - 0.5
    W2 = np.random.rand(300, 500) - 0.5
    b2 = np.random.rand(300, 1) - 0.5
    W3 = np.random.rand(10, 300) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3 

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y =  one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, W3, b3, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2 
    W3 = W3 - alpha * dW3  
    b3 = b3 - alpha * db3     
    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    # Wh1,Wh2 = corenet(epsilon, delta, data)
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2,W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, W3, b3, dW3, db3, alpha)
        if i % 10 == 0 or i==iterations-1:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3

def find_del(S,W):
    epsilon = 1e-10  
    D = np.zeros((W.shape[0], S.shape[1]))
    print("finding del")
    print(W.shape)
    for x in range(S.shape[1]):
        col = S[:, x]
        D_x=0
        for i in range(W.shape[0]):
            D_num = []
            D_den = []
            for j in range(W.shape[1]):
                prod = np.dot(W[i,j],col[j])
                D_den.append(prod)
                D_num.append(np.abs(prod))
            if np.abs(np.sum(D_den)) > epsilon:
                D_x = sum(D_num)/np.abs(sum(D_den))
            D[i,x] = D_x
    Dh = max(np.sum(D, axis=1)) 
    return D,Dh

def call_sparsify(W,W_plus_rv,W_minus_rv,W_rvs,epsilonL,delta,X_train,S,A,n):
    Wh = np.zeros_like(W)
    for i in range(W.shape[0]):
        W_rv_plus = sparsify(W_plus_rv[i],W_rvs[i],epsilonL,delta,X_train,S,A,n)
        print("+sparsification " , i)
        W_rv_minus = sparsify(W_minus_rv[i],W_rvs[i],epsilonL,delta,X_train,S,A,n)
        print("-sparsification ", i)
        W_rv = W_rv_plus - W_rv_minus
        Wh[i] = W_rv[0]
    return Wh


def corenet(epsilon, delta, data,W1, b1, W2, b2,W3,b3):
    num_neurons = [784,500,300,10]
    theta = [W1,W2,W3]
    L = len(theta)
    epsilon2 = epsilon/(2*(L-1))
    n_star = max(num_neurons)
    n = sum(num_neurons)
    num = int(n*n_star)
    print(n_star)
    print(n)
    print(num)
    lambda_star = float(math.log(num,10)/2)
    print(lambda_star)
    k = 2200
    sample_size = int(np.round(k*math.log(8*n*n_star/delta)))
    print(sample_size)
    print(data.shape)
    data = data.T
    sampled_set = np.random.choice(data.shape[1],size=sample_size,replace=False)
    S = data[:,sampled_set]
    print(data.shape)
    print(S.shape)
    m, n = S.shape
    data = data.T
    # S = S.T
    # print(W1.shape)
    Z1 = W1.dot(S) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)

    # print(A1.shape)
    # print(A2.shape)
    # print(Z1.shape)
    # print(Z2.shape)
    D1,Dh1 = find_del(S,W1)
    D2, Dh2 = find_del(A1,W2)
    D3, Dh3 = find_del(A2,W3)
    k = math.sqrt(2 * lambda_star)*(1 + math.sqrt(2 * lambda_star) * math.log(8*n*n_star/delta))
    print(k)
    Dh1 = Dh1/S.shape[1] + k
    Dh2 = Dh2/S.shape[1] + k
    Dh3 = Dh3/S.shape[1] + k
    Dh = Dh1*Dh2*Dh3
    epsilonL = epsilon2/Dh

    W1_plus = np.zeros_like(W1)
    W1_minus = np.zeros_like(W1)
    W1_plus = np.maximum(W1, 0)
    W1_minus = np.minimum(W1, 0)
   
    W2_plus = np.zeros_like(W2)
    W2_minus = np.zeros_like(W2)
    W2_plus = np.maximum(W2, 0)
    W2_minus = np.minimum(W2, 0)

    W3_plus = np.zeros_like(W3)
    W3_minus = np.zeros_like(W3)
    W3_plus = np.maximum(W3, 0)
    W3_minus = np.minimum(W3, 0)

    W1_plus_rv = [row.reshape(1, -1) for row in W1_plus]
    W2_plus_rv = [row.reshape(1, -1) for row in W2_plus]
    W3_plus_rv = [row.reshape(1, -1) for row in W3_plus]
    W1_minus_rv = [row.reshape(1, -1) for row in W1_minus]
    W2_minus_rv = [row.reshape(1, -1) for row in W2_minus]
    W3_minus_rv = [row.reshape(1, -1) for row in W3_minus]
    W1_rvs = [row.reshape(1, -1) for row in W1]
    W2_rvs = [row.reshape(1, -1) for row in W2]
    W3_rvs = [row.reshape(1, -1) for row in W3]

    Wh1 = call_sparsify(W1,W1_plus_rv,W1_minus_rv,W1_rvs,epsilonL,delta,X_train,S,S,n)
    Wh2 = call_sparsify(W2,W2_plus_rv,W2_minus_rv,W2_rvs,epsilonL,delta,X_train,S,A1,n)
    Wh3 = call_sparsify(W3,W3_plus_rv,W3_minus_rv,W3_rvs,epsilonL,delta,X_train,S,A2,n)
    
    return Wh1,Wh2,Wh3
 
def sparsify(W,W_rvs,epsilonL,delta,X_train,S,A,n):
    print("sparsification")
    all_prods = []
    for x in range(A.shape[1]):
        col_prods = []
        for j in range(W.shape[1]):
            col_prods.append(np.dot(W[0,j],A[j,x]))
            # print(np.dot(W[0,j],A[j,x]))
        all_prods.append(col_prods)
    sums = []
    for i in all_prods:
        sums.append(sum(i))
    s_j = []
    print(W.shape[1])
    for j in range(W.shape[1]):
        temp = []
        for x in range(A.shape[1]):
            temp.append(all_prods[x][j]/sums[x])
        s_j.append(max(temp))
    # print("sj", s_j)
    St = sum(s_j)
    print("St",St)
    q_j=[]
    for j in range(W.shape[1]):
        q_j.append(s_j[j]/St)
    
    K = 0.1
    # print(q_j)
    m = math.ceil((8*int(St)*K*math.log(8*n/delta))/epsilon*epsilon)
    # print(m)
    # nan_mask = np.isnan(q_j)
    # q_j[nan_mask] = 0
    # q_j = q_j / np.sum(q_j)
    # m = np.count_nonzero(q_j)
    # m=130
    print(m)

    sampled_indices = np.random.choice(np.arange(W.shape[1]), size=m, p=q_j, replace=True)
    # print(sampled_indices)
    wh = np.zeros((1, W.shape[1]))

    for j in sampled_indices:
        wh[0, j] += W[0, j] / (m * q_j[j])

    print(wh.shape)
    return wh


def gd2(X, Y, alpha, iterations,W1,W2,b1,b2,w3,b3):
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2,W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, W3, b3, dW3, db3, alpha)
        if i % 10 == 0 or i==iterations-1:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    # plt.imshow(current_image)
    # plt.show()
    plt.imshow(current_image, cmap='gray')
    plt.show()
    plt.savefig('image.png') 

print(X_train.shape)
print(X_dev.shape)
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train.T, Y_train, 0.1, 700)
print("test")
dev_predictions = make_predictions(X_dev.T, W1, b1, W2, b2, W3, b3)

print(get_accuracy(dev_predictions, Y_dev))
Wh1,Wh2,Wh3 = corenet(epsilon, delta, X_train,W1, b1, W2, b2, W3, b3)

# W1, b1, W2, b2, W3, b3 = gd2(X_train.T, Y_train, 0.10, 500, Wh1,Wh2,Wh3,b1,b2,b3)


print("test")
dev_predictions = make_predictions(X_dev.T, Wh1, b1, Wh2, b2, Wh3, b3)
print(get_accuracy(dev_predictions, Y_dev))
