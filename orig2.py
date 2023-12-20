import math
import numpy as np
# import cupy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('/home/shubham/gurmehak/DDcoresets/train.csv', low_memory=False)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:10000].T
Y_dev = torch.tensor(data_dev[0], dtype=torch.float32)
X_dev = torch.tensor(data_dev[1:], dtype=torch.float32) / 255

data_train = data[10000:].T
Y_train = torch.tensor(data_train[0], dtype=torch.float32)
X_train = torch.tensor(data_train[1:], dtype=torch.float32) / 255

# Move tensors to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Y_dev = Y_dev.to(device)
X_dev = X_dev.to(device)
Y_train = Y_train.to(device)
X_train = X_train.to(device)

# Get the dimensions
_, m_train = X_train.shape

print(X_train.shape)
print(X_dev.shape)

epsilon = 0.5
delta = 0.5


def init_params():
    W1 = torch.rand(10, 784, device=device) - 0.5
    b1 = torch.rand(10, 1, device=device) - 0.5
    W2 = torch.rand(10, 10, device=device) - 0.5
    b2 = torch.rand(10, 1, device=device) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return torch.relu(Z)

def softmax(Z):
    A = torch.nn.functional.softmax(Z.clone().detach().to(torch.float32), dim=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    X = X.clone().detach()
    W1 = W1.clone().detach()
    b1 = b1.clone().detach()
    W2 = W2.clone().detach()
    b2 = b2.clone().detach()

    Z1 = torch.matmul(W1, X) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = torch.nn.functional.softmax(Z2, dim=0)

    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    # one_hot_Y = torch.zeros((Y.size(0), int(Y.max() + 1)), dtype=torch.float32)
    # one_hot_Y[torch.arange(Y.size(0), dtype=torch.long), Y] = 1
    # one_hot_Y = one_hot_Y.T
    one_hot_Y = F.one_hot(Y.long(), num_classes=int(Y.max() + 1)).T.float()
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * torch.matmul(dZ2, A1.T)
    db2 = 1 / m * torch.sum(dZ2)
    dZ1 = torch.matmul(W2.T, dZ2) * (Z1 > 0).float()  # ReLU derivative in PyTorch
    dW1 = 1 / m * torch.matmul(dZ1, X.T)
    db1 = 1 / m * torch.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * torch.tensor(dW1, dtype=torch.float32)
    b1 = b1 - alpha * torch.tensor(db1, dtype=torch.float32)
    W2 = W2 - alpha * torch.tensor(dW2, dtype=torch.float32)
    b2 = b2 - alpha * torch.tensor(db2, dtype=torch.float32)
    return W1, b1, W2, b2


def get_predictions(A2):
    return torch.argmax(A2.clone().detach(), dim=0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return torch.sum(predictions.clone().detach() == Y.clone().detach()) / len(Y)


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0 or i==iterations-1:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def find_del(S,W):
    epsilon = 1e-10
    D = torch.zeros((W.shape[0], S.shape[1]))
    print("in del",D.shape)
    for x in range(S.shape[1]):
        col = S[:, x]
        D_x = 0
        for i in range(W.shape[0]):
            print(i,x)
            D_num = torch.tensor([],device=device)
            D_den = torch.tensor([],device=device)

            for j in range(W.shape[1]):
                products = torch.multiply(W[i,j], col)
                D_den = torch.cat((D_den, products))
                D_num = torch.cat((D_num, torch.abs(products)))

            if torch.abs(torch.sum(D_den)) > epsilon:
                D_x = torch.sum(D_num) / torch.abs(torch.sum(D_den))

            D[i, x] = D_x

    Dh = torch.max(torch.sum(D, dim=1))
    return D, Dh


def corenet(epsilon, delta, data, W1, b1, W2, b2):
    num_neurons = [784, 10, 10]
    theta = [W1, W2]
    L = len(theta)
    epsilon2 = epsilon / (2 * (L - 1))
    n_star = max(num_neurons)
    n = sum(num_neurons)
    num = int(n * n_star)
    print(n_star)
    print(n)
    print(num)
    lambda_star = float(math.log(num, 10) / 2)
    print(lambda_star)
    k = 1000
    sample_size = int(np.round(k * math.log(8 * n * n_star / delta)))
    print(sample_size)
    sampled_set = np.random.choice(data.shape[1], size=sample_size, replace=False)
    S = data[:, sampled_set]
    print(data.shape)
    print(S.shape)
    m, n = S.shape
    data = data.T

    Z1 = torch.matmul(W1, S) + b1
    A1 = ReLU(Z1)
    Z2 = torch.matmul(W2, A1) + b2
    A2 = softmax(Z2)
    print(S.shape)
    print(W1.shape)
    D1, Dh1 = find_del(S, W1)
    print(S.shape)
    print(W1.shape)
    print("d1 done")
    D2, Dh2 = find_del(A1, W2)
    print(A1.shape)
    print(W2.shape)
    print("d2 done")

    k = math.sqrt(2 * lambda_star) * (1 + math.sqrt(2 * lambda_star) * math.log(8 * n * n_star / delta))
    print(k)
    Dh1 = Dh1 / S.shape[1] + k
    Dh2 = Dh2 / S.shape[1] + k
    Wh1 = torch.zeros_like(W1)
    Wh2 = torch.zeros_like(W2)
    Dh = Dh1 * Dh2
    epsilonL = epsilon2 / Dh
    W1_plus = torch.zeros_like(W1)
    W1_minus = torch.zeros_like(W1)
    W1_plus = ReLU(W1)
    W1_minus = torch.min(W1, torch.zeros_like(W1))

    W2_plus = torch.zeros_like(W2)
    W2_minus = torch.zeros_like(W2)
    print(torch.count_nonzero(W2))
    W2_plus = ReLU(W2)
    print(torch.count_nonzero(W2_plus))
    W2_minus = torch.min(W2, torch.zeros_like(W2))
    W1_rvs = [row.reshape(1, -1) for row in W1]
    W2_rvs = [row.reshape(1, -1) for row in W2]

    for i in range(W1.shape[0]):
        W1_rv_plus = sparsify(W1_plus[i].reshape(1, -1), W1_rvs[i], epsilonL, delta, X_train, S, S, n)
        print("+sparsification", i)
        W1_rv_minus = sparsify(W1_minus[i].reshape(1, -1), W1_rvs[i], epsilonL, delta, X_train, S, S, n)
        print("-sparsification", i)
        Wh1[i] = W1_rv_plus - W1_rv_minus

    for i in range(W2.shape[0]):
        W2_rv_plus = sparsify(W2_plus[i].reshape(1, -1), W2_rvs[i], epsilonL, delta, X_train, S, A1, n)
        print("+sparsification ", i)
        W2_rv_minus = sparsify(W2_minus[i].reshape(1, -1), W2_rvs[i], epsilonL, delta, X_train, S, A1, n)
        print("-sparsification ", i)
        Wh2[i] = W2_rv_plus - W2_rv_minus

    return Wh1, Wh2

def sparsify(W, W_rvs, epsilonL, delta, X_train, S, A, n):
    print("sparsification")
    all_prods = []
    for x in range(A.shape[1]):
        col_prods = []
        for j in range(W.shape[1]):
            col_prods.append(torch.mul(W[0, j], A[j, x]))
        col_prods_tensor = torch.stack(col_prods)

        all_prods.append(col_prods_tensor)

    all_prods = torch.stack(all_prods, dim=0)
    print(all_prods.shape)
    sums = torch.sum(all_prods)
    sums = []
    for i in all_prods:
        sums.append(sum(i.cpu().numpy()))
    print(len(sums))
    sums = torch.tensor(sums, device=device)
    s_j = []
    print(W.shape[1])
    print(A.shape[1])
    for j in range(W.shape[1]):
        temp = []
        for x in range(A.shape[1]):
            temp.append(all_prods[x][j]/sums[x])
  
        s_j.append(max(temp))
        
    St = sum(s_j)
    print("St", St)
    
    q_j = torch.tensor([s_j[j] / St for j in range(W.shape[1])],device=device)
    
    K = 0.1
    m = math.ceil((8 * int(St) * K * math.log(8 * n / delta)) / epsilonL * epsilonL)
    # print(m)
    
    # m = torch.count_nonzero(q_j)
    # m = 0.6*m
    print(m)
    
    m = m.to(device)
    sampled_indices = torch.multinomial(q_j.clone().detach(), int(m), replacement=True)
    wh = torch.zeros((1, W.shape[1]),device=device)
    
    for j in sampled_indices:
        wh[0, j] += W[0, j] / (m * q_j[j])
    
    print(wh.shape)
    return wh

def gd2(X, Y, alpha, iterations,W1,W2,b1,b2):
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0 or i==iterations-1:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    # current_image = current_image.cpu().numpy().reshape((28, 28)) * 255
    # plt.imshow(current_image, cmap='gray')  # Adjust the cmap as needed
    # plt.show()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 700)
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))

Wh1, Wh2 = corenet(epsilon, delta, X_train, W1, b1, W2, b2)
print("corenet done")
print(W1.shape)
print(Wh1.shape)
print(W2.shape)
print(Wh2.shape)

# W1, b1, W2, b2 = gd2(X_train, Y_train, 0.10, 500, Wh1, Wh2, b1, b2)
# test_prediction(0, W1, b1, W2, b2)
# test_prediction(1, W1, b1, W2, b2)
# test_prediction(2, W1, b1, W2, b2)
# test_prediction(3, W1, b1, W2, b2)

dev_predictions = make_predictions(X_dev, Wh1, b1, Wh2, b2)
print(get_accuracy(dev_predictions, Y_dev))
