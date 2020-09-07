import numpy as np
from numpy import linalg as LA
import gzip
import pickle

def imload(mode, N, a, b, m, n, offset):
    with gzip.open('mnist.pkl.gz', 'rb') as mnist:
        train_set, valid_set, test_set = pickle.load(mnist, encoding='latin1')

    imagevectors = np.zeros((N, m*n))
    images = np.zeros((N, m, n))
    if mode == 'train':
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = train_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
        identity = np.zeros((N))
        identity = train_set[1][0:N].reshape((N))
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id
    if mode == 'trainvalid':
        imagevectors = np.zeros((N + 10000, m * n))
        images = np.zeros((N + 10000, m, n))
        onehot_id = np.zeros((N + 10000, 10))
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = train_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
        for i in range(N, N + 10000):
            images[i][offset:offset + a, offset:offset + b] = valid_set[0][i - N].reshape((a, b))
            imagevectors[i] = images[i].reshape((m * n))
        identity = np.zeros((N + 10000), dtype=np.int32)
        identity[:N] = train_set[1][0:N].reshape((N))
        identity[N:] = valid_set[1].reshape((10000))
        N += 10000
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id
    if mode == 'test':
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = test_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
        identity = np.zeros((N))
        identity = test_set[1][0:N].reshape((N))
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id

def imloadq(mode, N, a, b, m, n, offset):
    with gzip.open('mnist.pkl.gz', 'rb') as mnist:
        train_set, valid_set, test_set = pickle.load(mnist, encoding='latin1')

    f = gzip.open('qmnist-test-images-idx3-ubyte.gz', 'r')



    f.read(16)
    buf = f.read(a * b * N)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(N, a, b, 1) / 256.0

    fl = gzip.open('qmnist-test-labels-idx1-ubyte.gz')

    fl.read(8)
    bufl = fl.read(N)
    datal = np.frombuffer(bufl, dtype=np.uint8).astype(np.int32)
    datal = datal.reshape(-1)

    imagevectors = np.zeros((N, m*n))
    images = np.zeros((N, m, n))
    if mode == 'train':
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = train_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
        identity = np.zeros((N))
        identity = train_set[1][0:N].reshape((N))
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id
    if mode == 'trainvalid':
        imagevectors = np.zeros((N + 10000, m * n))
        images = np.zeros((N + 10000, m, n))
        onehot_id = np.zeros((N + 10000, 10))
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = train_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
        for i in range(N, N + 10000):
            images[i][offset:offset + a, offset:offset + b] = valid_set[0][i - N].reshape((a, b))
            imagevectors[i] = images[i].reshape((m * n))
        identity = np.zeros((N + 10000), dtype=np.int32)
        identity[:N] = train_set[1][0:N].reshape((N))
        identity[N:] = valid_set[1].reshape((10000))
        N += 10000
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id
    if mode == 'test':
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = data[i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
        identity = np.zeros((N))
        identity = datal[0:N].reshape((N))
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id


def imload_L2mean(mode, N, a, b, m, n, offset, noise_lvl, scale=1):
    #mnist_L2mean = 9.17869424796783
    mnist_L2mean = scale
    with gzip.open('mnist.pkl.gz', 'rb') as mnist:
        train_set, valid_set, test_set = pickle.load(mnist, encoding='latin1')

    imagevectors = np.zeros((N, m*n))
    images = np.zeros((N, m, n))
    if mode == 'train':
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = train_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
            imagevectors[i] = imagevectors[i] * mnist_L2mean / LA.norm(imagevectors[i])
            images[i] = imagevectors[i].reshape((a, b))#images[i] * mnist_L2mean / LA.norm(imagevectors[i])
        identity = np.zeros((N))
        identity = train_set[1][0:N].reshape((N))
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        return imagevectors, images, identity, onehot_id
    if mode == 'trainvalid':
        imagevectors = np.zeros((N + 10000, m * n))
        images = np.zeros((N + 10000, m, n))
        onehot_id = np.zeros((N + 10000, 10))
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = train_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
            imagevectors[i] = imagevectors[i] * mnist_L2mean / LA.norm(imagevectors[i])
            images[i] = imagevectors[i].reshape((m, n))#images[i] * mnist_L2mean / LA.norm(imagevectors[i])

        for i in range(N, N + 10000):
            images[i][offset:offset + a, offset:offset + b] = valid_set[0][i - N].reshape((a, b))
            imagevectors[i] = images[i].reshape((m * n))
            imagevectors[i] = imagevectors[i] * mnist_L2mean / LA.norm(imagevectors[i])
            images[i] = imagevectors[i].reshape((m, n))#images[i] * mnist_L2mean / LA.norm(imagevectors[i])

        identity = np.zeros((N + 10000), dtype=np.int32)
        identity[:N] = train_set[1][0:N].reshape((N))
        identity[N:] = valid_set[1].reshape((10000))
        N += 10000
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        noise = np.random.normal(0, noise_lvl, (N, m*n))
        imagevectors += noise
        images += noise.reshape((N, m, n))
        return imagevectors, images, identity, onehot_id
    if mode == 'test':
        for i in range(N):
            images[i][offset:offset+a, offset:offset+b] = test_set[0][i].reshape((a, b))
            imagevectors[i] = images[i].reshape((m*n))
            imagevectors[i] = imagevectors[i] * mnist_L2mean / LA.norm(imagevectors[i])
            images[i] = imagevectors[i].reshape((m, n))#images[i] * mnist_L2mean / LA.norm(imagevectors[i])
        identity = np.zeros((N))
        identity = test_set[1][0:N].reshape((N))
        onehot_id = np.zeros((N, 10))
        for i in range(N):
            onehot_id[i, identity[i]] = 1
        noise = np.random.normal(0, noise_lvl, (N, m*n))
        imagevectors += noise
        images += noise.reshape((N, m, n))
        return imagevectors, images, identity, onehot_id