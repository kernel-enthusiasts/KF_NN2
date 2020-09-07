from __future__ import absolute_import, division, print_function, unicode_literals
#from import_mnist import imload
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
#from tensorflow._api.v1.keras import datasets, layers, models
import time
import numpy.linalg as LA
import matplotlib.pyplot as plt
import copy
from import_mnist import imload
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
#tf.random.set_random_seed(0)
#np.random.seed(0)
s = 0

datagen = IDG(rotation_range=10, zoom_range=0.03, width_shift_range=2, height_shift_range=2, shear_range=10)
#datagen = IDG(rotation_range = 10, zoom_range = 0.10, width_shift_range = 0.1, height_shift_range = 0.1)

tr_vec, tr_im, tr_id, tr_ohid1 = imload('trainvalid', 50000, 28, 28, 28, 28, 0)
te_vec = tr_vec[50000:]
te_im = tr_im[50000:]
te_id = tr_id[50000:]
te_ohid1 = tr_ohid1[50000:]

tr_vec = tr_vec[:50000]
tr_im = tr_im[:50000]
tr_id = tr_id[:50000]
tr_ohid1 = tr_ohid1[:50000]

tr_ohid = tr_ohid1 - 0.1
te_ohid = te_ohid1 - 0.1

N_tr = 50000


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(0)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dz = np.zeros_like(dx)

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)

num_traindigits = np.sum(tr_ohid1, axis=0)
num_traindigits = num_traindigits.astype(np.int32)
digits = {}
for i in range(10):
    digits[i] = []
digit_list = np.zeros((6742, 10))
counter_list = np.zeros((10), dtype=np.int32)
for i in range(N_tr):
    digit_i = int(tr_id[i])
    digits[digit_i].append(i)
    digit_list[counter_list[digit_i], digit_i] = i
    counter_list[digit_i] += 1

def get_rand_batch4(batch_size):
    #batch_size a mult of 20
    test_indx = np.zeros((batch_size), dtype=np.int32)
    digit_size = batch_size // 20
    for i in range(10):
        #digit_set = np.random.choice(num_traindigits[i], 2 * digit_size, replace=False)
        #digit_indx = digit_list[digit_set, i].reshape((-1))
        digit_indx = np.random.choice(digits[i], 2 * digit_size, replace=False)
        test_indx[i * digit_size: (i + 1) * digit_size] = digit_indx[:digit_size]
        test_indx[(i + 10) * digit_size: (i + 11) * digit_size] = digit_indx[digit_size:]
    batch_X = tr_im[test_indx]
    batch_Y = tr_ohid[test_indx]

    for indx in range(batch_size):
        state = np.random.randint(6, size=1)[0]
        if state == 1 or state == 2:
            def_im = datagen.flow(batch_X[indx].reshape((1, 28, 28, 1)), batch_size=1)
            batch_X[indx] = def_im[0][0].reshape((28, 28))
        if state == 3 or state == 4:
            batch_X[indx] = elastic_transform(batch_X[indx].reshape((28, 28)), 150, 10)
    return batch_X, batch_Y

def get_rand_batch40(batch_size):
    #batch_size a mult of 20
    test_indx = np.zeros((batch_size), dtype=np.int32)
    digit_size = batch_size // 20
    for i in range(10):
        digit_set = np.random.choice(num_traindigits[i], 2 * digit_size, replace=False)
        digit_indx = digit_list[digit_set, i].reshape((-1))
        test_indx[i * digit_size: (i + 1) * digit_size] = digit_indx[:digit_size]
        test_indx[(i + 10) * digit_size: (i + 11) * digit_size] = digit_indx[digit_size:]
    batch_X = tr_im[test_indx]
    batch_Y = tr_ohid[test_indx]

    return batch_X, batch_Y

def CNN_KF18bn_RBF_AUG(lrate, drate):
    # Uses all 60K MNIST training points
    # CNN with KF training with rbf kernel
    #tf.random.set_random_seed(0)
    #np.random.seed(0)
    X0 = tf.placeholder(tf.float32, [None, 28, 28])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    pkeep2 = tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)

    ba_size = tf.placeholder(tf.float32)

    X = tf.reshape(X0, [-1, 28, 28, 1])
    # Y_ = tf.one_hot(Y_id, 10)

    H = 150
    I = 150
    J = 150
    K = 300
    L = 300
    M = 300
    full_layer_size = 300
    N2 = 1200
    N = full_layer_size
    width = N
    in_fac = 1
    bias_init = 0.01

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.9,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    #tf.random.set_random_seed(0)
    W1 = tf.Variable(tf.random.uniform([3, 3, 1, H],
                                       minval=-tf.sqrt(6 / (9 * (1 + H))),
                                       maxval=tf.sqrt(6 / (9 * (1 + H)))))
    B1 = tf.Variable(tf.constant(bias_init, tf.float32, [H]))
    W2 = tf.Variable(tf.random.uniform([3, 3, H, I],
                                       minval=-tf.sqrt(6 / (9 * (H + I))),
                                       maxval=tf.sqrt(6 / (9 * (H + I)))))
    B2 = tf.Variable(tf.constant(bias_init, tf.float32, [I]))
    W3 = tf.Variable(tf.random.uniform([5, 5, I, J],
                                       minval=-tf.sqrt(6 / (25 * (I + J))),
                                       maxval=tf.sqrt(6 / (25 * (I + J)))))
    B3 = tf.Variable(tf.constant(bias_init, tf.float32, [J]))
    W4 = tf.Variable(tf.random.uniform([3, 3, J, K],
                                       minval=-tf.sqrt(6 / (9 * (J + K))),
                                       maxval=tf.sqrt(6 / (9 * (J + K)))))
    B4 = tf.Variable(tf.constant(bias_init, tf.float32, [K]))
    W5 = tf.Variable(tf.random.uniform([3, 3, K, L],
                                       minval=-tf.sqrt(6 / (9 * (K + L))),
                                       maxval=tf.sqrt(6 / (9 * (K + L)))))
    B5 = tf.Variable(tf.constant(bias_init, tf.float32, [L]))
    W6 = tf.Variable(tf.random.uniform([5, 5, L, M],
                                       minval=-tf.sqrt(6 / (25 * (L + M))),
                                       maxval=tf.sqrt(6 / (25 * (L + M)))))
    B6 = tf.Variable(tf.constant(bias_init, tf.float32, [M]))
    # W15 = tf.Variable(tf.random.uniform([28 * 28 * J, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 28 * 28 * J)),
    #                                   maxval=tf.sqrt(0.01 / (N + 28 * 28 * J))))
    # W25 = tf.Variable(tf.random.uniform([14 * 14 * K, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * K)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * K))))
    # W35 = tf.Variable(tf.random.uniform([14 * 14 * L, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * L)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * L))))
    W7 = tf.Variable(tf.random.uniform([16 * M, N],
                                       minval=-tf.sqrt(6 / (N + 16 * M)),
                                       maxval=tf.sqrt(6 / (N + 16 * M))))
    B7 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))

    W8 = tf.Variable(tf.random.uniform([N, N2], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B8 = tf.Variable(tf.constant(bias_init, tf.float32, [N2]))
    W9 = tf.Variable(tf.random.uniform([N2, N], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B9 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    # W58 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(0.01 / (N + 10)), maxval=tf.sqrt(0.01 / (N + 10))))
    # W68 = tf.Variable(tf.random.uniform([N2, 10], minval=-tf.sqrt(0.01 / (N2 + 10)), maxval=tf.sqrt(0.01 / (N2 + 10))))

    W10 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(6 / (N + 10)), maxval=tf.sqrt(6 / (N + 10))))
    B10 = tf.Variable(tf.constant(bias_init, tf.float32, [10]))

    W37 = tf.Variable(tf.random.uniform([144 * J, N],
                                        minval=-tf.sqrt(6 / 144 * J + N),
                                        maxval=tf.sqrt(6 / 144 * J + N)))
    B37 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    s37 = tf.Variable(tf.constant(1, tf.float32, [1, J]))



    s = tf.Variable(tf.constant([2, 2], tf.float32, [2]))

    # stride = 1  # output is 28x28
    # Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    # stride = 2  # output is 14x14
    # Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    # stride = 2  # output is 7x7
    # Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    stride = 1  # output is 28x28
    # Y0 = tf.nn.relu(tf.nn.conv2d(X, W0, strides=[1, stride, stride, 1], padding='SAME') + B0)

    Y1a = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + B1
    Y1b, update_ema1 = batchnorm(Y1a, tst, step, B1, convolutional=True)
    Y1c = tf.nn.relu(Y1b)
    Y1 = tf.nn.dropout(Y1c, pkeep)

    Y2a = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2
    Y2b, update_ema2 = batchnorm(Y2a, tst, step, B2, convolutional=True)
    Y2c = tf.nn.relu(Y2b)
    Y2 = tf.nn.dropout(Y2c, pkeep)

    Y3a = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3b, update_ema3 = batchnorm(Y3a, tst, step, B3, convolutional=True)
    Y3c = tf.nn.relu(Y3b)
    Y3d = tf.nn.dropout(Y3c, pkeep)
    Y3 = tf.nn.max_pool2d(Y3d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    Y4a = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4
    Y4b, update_ema4 = batchnorm(Y4a, tst, step, B4, convolutional=True)
    Y4c = tf.nn.relu(Y4b)
    Y4 = tf.nn.dropout(Y4c, pkeep)

    Y5a = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5
    Y5b, update_ema5 = batchnorm(Y5a, tst, step, B5, convolutional=True)
    Y5c = tf.nn.relu(Y5b)
    Y5 = tf.nn.dropout(Y5c, pkeep)

    Y6a = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6
    Y6b, update_ema6 = batchnorm(Y6a, tst, step, B6, convolutional=True)
    Y6c = tf.nn.relu(Y6b)
    Y6d = tf.nn.dropout(Y6c, pkeep)
    Y6 = tf.nn.max_pool2d(Y6d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    # reshape the output from the third convolution for the fully connected layer
    # YY1 = tf.reshape(Y1, shape=[-1, 24 * 24 * J])
    # YY2 = tf.reshape(Y2, shape=[-1, 12 * 12 * K])
    # YY3 = tf.reshape(Y3, shape=[-1, 12 * 12 * L])
    Y7 = tf.reduce_mean(tf.reshape(Y6, shape=[-1, 16, M]), axis=1)
    #Y7 = tf.reduce_mean(tf.reshape(Y6d, shape=[-1, 64, M]), axis=1)

    # Y5 = tf.matmul(YY1, W15) + tf.matmul(YY2, W25) + tf.matmul(YY3, W35) + tf.matmul(YY, W5) + B5
    #Y7 = tf.matmul(YY, W7) + B7
    Y37 = tf.reduce_mean(tf.reduce_mean(Y3, axis=1), axis=1)#tf.matmul(tf.reshape(Y3, shape=[-1, 144 * J]), W37) + B37



    Y8a = tf.matmul(Y7, W8) + B8
    Y8b, update_ema8 = batchnorm(Y8a, tst, step, B8, convolutional=False)
    Y8c = tf.nn.relu(Y8b)
    Y8 = tf.nn.dropout(Y8c, pkeep2)

    Y9a = tf.matmul(Y8, W9) + B9
    Y9b, update_ema9 = batchnorm(Y9a, tst, step, B9, convolutional=False)
    Y9c = tf.nn.relu(Y9b)
    Y9 = tf.nn.dropout(Y9c, pkeep2)

    Ylogits = tf.matmul(Y9, W10) + B10

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema8,
                          update_ema9)

    s = tf.maximum(s, 0.3)

    #kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    #weight = s[0] / tf.reduce_mean(kernel0)
    #kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sim_mat = tf.reduce_sum(tf.reshape(Y_ + 0.1, (-1, 1, 10)) * tf.reshape(Y_ + 0.1, (1, -1, 10)), axis=2)
    all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
    same_mat = sim_mat * all_mat
    diff_mat = all_mat * (1 - sim_mat)

    # kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    # weight = s[0] / tf.reduce_mean(kernel0)
    # kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sq_YY0 = tf.reduce_sum(tf.square(Y37), axis=1)
    sq_YY0 = tf.reshape(sq_YY0, (-1, 1))

    pdist0 = -2 * tf.matmul(Y37, tf.transpose(Y37))
    pdist0 = pdist0 + sq_YY0
    pdist0 = pdist0 + tf.transpose(sq_YY0)
    kernel0 = tf.exp(-s[0] * pdist0 / tf.reduce_mean(pdist0))

    sq_YY = tf.reduce_sum(tf.square(Y7), axis=1)
    sq_YY = tf.reshape(sq_YY, (-1, 1))

    pdist = -2 * tf.matmul(Y7, tf.transpose(Y7))
    pdist = pdist + sq_YY
    pdist = pdist + tf.transpose(sq_YY)
    kernel = tf.exp(-s[1] * pdist / tf.reduce_mean(pdist))

    presig = tf.range(tf.shape(Y_)[0])
    # presig = tf.random_shuffle(presig)
    sig = presig[:tf.shape(Y_)[0] // 2]  # Gets the indicies of a random sample of half of the batch/data
    sig1 = presig[tf.shape(Y_)[0] // 2:]

    pi = tf.one_hot(sig, tf.shape(Y_)[0])
    pi1 = tf.one_hot(sig1, tf.shape(Y_)[0])

    smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    smallkernel1 = tf.matmul(pi1, tf.matmul(kernel, tf.linalg.matrix_transpose(pi1)))

    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff = tf.linalg.solve(smallkernel, Y_small)
    coeff1 = tf.linalg.solve(smallkernel1, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
    test_kernel = kernel[:, :tf.shape(Y_)[0] // 2]
    test_kernel1 = kernel[:, tf.shape(Y_)[0] // 2:]

    net_class = tf.matmul(test_kernel, coeff)
    net_class1 = tf.matmul(test_kernel1, coeff1)

    test_class = tf.matmul(pi_full, Y_)

    class_error = tf.reduce_sum(tf.square(net_class - test_class))
    class_error += tf.reduce_sum(tf.square(net_class1 - test_class))

    norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

    ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
    norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

    rho2 = 1 - norm_sig / norm_dagger

    smallkernel0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    smallkernel01 = tf.matmul(pi1, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi1)))
    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff0 = tf.linalg.solve(smallkernel0, Y_small)
    coeff01 = tf.linalg.solve(smallkernel01, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel0 = tf.matmul(pi_full, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi_full)))
    test_kernel0 = kernel0[:, :tf.shape(Y_)[0] // 2]
    test_kernel01 = kernel0[:, tf.shape(Y_)[0] // 2:]

    net_class0 = tf.matmul(test_kernel0, coeff0)
    net_class01 = tf.matmul(test_kernel01, coeff01)
    test_class0 = tf.matmul(pi_full, Y_)

    class_error0 = tf.reduce_sum(tf.square(net_class0 - test_class0)) + tf.reduce_sum(
        tf.square(net_class01 - test_class0))

    norm_dagger0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel0, Y_)))

    ker_bar0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    A_til0 = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar0, pi))
    norm_sig0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til0, Y_)))

    rho20 = 1 - norm_sig0 / norm_dagger0

    same_dist = tf.reduce_sum(pdist * same_mat) / tf.reduce_sum(same_mat)
    diff_dist = tf.reduce_sum(pdist * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist = tf.reduce_sum(pdist * all_mat) / tf.reduce_sum(all_mat)
    same_dist0 = tf.reduce_sum(pdist0 * same_mat) / tf.reduce_sum(same_mat)
    diff_dist0 = tf.reduce_sum(pdist0 * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist0 = tf.reduce_sum(pdist0 * all_mat) / tf.reduce_sum(all_mat)

    dists = [same_dist, diff_dist, all_dist, same_dist0, diff_dist0, all_dist0]


    # mse = tf.reduce_mean(tf.square(Ylogits - Y_))
    mse = tf.reduce_mean(tf.square(Ylogits - (Y_ + 0.1)))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits + 0.1, labels=(Y_ + 0.1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0000001 + tf.train.exponential_decay(lrate, step, drate, 1 / math.e)  # cut lr in future
    train_step = tf.train.AdamOptimizer(lr).minimize(0.0 * class_error0 + 0.0 * class_error + cross_entropy)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(10 * lr, 0.9).minimize(0.1 * class_error + cross_entropy)
    train_step1 = tf.train.AdamOptimizer(lr).minimize(class_error, var_list=[W1, B1, W2, B2, W3, B3, W4, B4])
    train_step2 = tf.train.AdamOptimizer(lr).minimize(mse, var_list=[W5, B5, W6, B6, W7, B7])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    acc_max = 0
    a = 0.3
    b = 0.3
    r = 0.1
    bs = 100
    err = np.zeros((121, 2, 2))
    rhoce = np.zeros((121, 3))
    distances = np.zeros((121, 6))
    acc = np.zeros((20))
    acct = np.zeros((120))

    for i in range(10001):
        # batch_X, batch_Y = get_batch(i)
        # if i < 5000: batch_X, batch_Y = get_rand_batch3(i, 600, tr_im, tr_ohid)
        # else: batch_X, batch_Y = get_rand_batch3(i, 100, tr_im, tr_ohid)
        # batch_size = int(150 + 2 * math.floor(375 * math.exp(-i / 1000.0)))
        # batch_size = 300

        # batch_X, batch_Y = get_def_rand_batch2(i, batch_size, tr_im, tr_ohid)
        batch_X, batch_Y = get_rand_batch4(bs)
        # for i in range(10):
        #    plt.imshow(batch_X[i].reshape((28, 28)))
        #    plt.show()
        # batch_X, batch_Y = get_rand_batch4(batch_size)

        sess.run(update_ema,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})
        sess.run(train_step,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})

        if i % 100 == 0:
            sig, rho1, ce1, ce0, cr1, ds = sess.run([s, rho2, class_error, class_error0, cross_entropy, dists],
                                                    {X0: batch_X, Y_: batch_Y, step: i, tst: False, pkeep: 1.0,
                                                     pkeep2: 1.0,
                                                     ba_size: bs})
            distances[i // 100] = np.asarray(ds)
            rhoce[i // 100, 0] = rho1
            rhoce[i // 100, 1] = ce0
            rhoce[i // 100, 2] = ce1
            print(i, "rho: ", rho1, " class error: ", ce0, ce1, "cross ent:", cr1, "sig:", sig, "dists:",
                  distances[i // 100], time.localtime())
        pkeep_i = 0.75  # + 0.25 * (math.exp(-(i - 0) / 1000))
        # if i < 5000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step2, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # if i < 3000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})

        if i % 1000 == 0 and i >= 10000:
            acc = np.zeros((20))
            c = np.zeros((20))
            for j in range(20):
                acc[j], c[j] = sess.run([accuracy, cross_entropy],
                                        {X0: te_im[j * 500: (j + 1) * 500],
                                         Y_: te_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                         ba_size: bs})

            err[i // 100, 0, 0] = 1 - acc.mean()
            err[i // 100, 0, 1] = c.mean()
            # a, c = sess.run([accuracy, cross_entropy], {X0: te_im, Y_id: te_id, pkeep: 1.0})
            if acc.mean() > acc_max:
                acc_max = acc.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* test accuracy:" + str(
                acc.mean()) + " test loss: " + str(c.mean()) + " max accuracy:" + str(acc_max))
        if i == 12000:
            acct = np.zeros((120))
            ct = np.zeros((120))
            for j in range(120):
                acct[j], ct[j] = sess.run([accuracy, cross_entropy],
                                          {X0: tr_im[j * 500: (j + 1) * 500],
                                           Y_: tr_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                           ba_size: bs})
            err[i // 100, 1, 0] = 1 - acct.mean()
            err[i // 100, 1, 1] = ct.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* train accuracy:" + str(
                acct.mean()) + " train loss: " + str(ct.mean()))

    return acc.mean(), acct.mean(), acc_max, err, rhoce, distances

def CNN_KF18do_RBF_AUG(pk, pk2):
    # Uses all 60K MNIST training points
    # CNN with KF training with rbf kernel
    X0 = tf.placeholder(tf.float32, [None, 28, 28])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    pkeep2 = tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)

    ba_size = tf.placeholder(tf.float32)

    X = tf.reshape(X0, [-1, 28, 28, 1])
    # Y_ = tf.one_hot(Y_id, 10)

    H = 150
    I = 150
    J = 150
    K = 300
    L = 300
    M = 300
    full_layer_size = 300
    N2 = 1200
    N = full_layer_size
    width = N
    in_fac = 1
    bias_init = 0.01

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.9,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    W1 = tf.Variable(tf.random.uniform([3, 3, 1, H],
                                       minval=-tf.sqrt(6 / (9 * (1 + H))),
                                       maxval=tf.sqrt(6 / (9 * (1 + H)))))
    B1 = tf.Variable(tf.constant(bias_init, tf.float32, [H]))
    W2 = tf.Variable(tf.random.uniform([3, 3, H, I],
                                       minval=-tf.sqrt(6 / (9 * (H + I))),
                                       maxval=tf.sqrt(6 / (9 * (H + I)))))
    B2 = tf.Variable(tf.constant(bias_init, tf.float32, [I]))
    W3 = tf.Variable(tf.random.uniform([5, 5, I, J],
                                       minval=-tf.sqrt(6 / (25 * (I + J))),
                                       maxval=tf.sqrt(6 / (25 * (I + J)))))
    B3 = tf.Variable(tf.constant(bias_init, tf.float32, [J]))
    W4 = tf.Variable(tf.random.uniform([3, 3, J, K],
                                       minval=-tf.sqrt(6 / (9 * (J + K))),
                                       maxval=tf.sqrt(6 / (9 * (J + K)))))
    B4 = tf.Variable(tf.constant(bias_init, tf.float32, [K]))
    W5 = tf.Variable(tf.random.uniform([3, 3, K, L],
                                       minval=-tf.sqrt(6 / (9 * (K + L))),
                                       maxval=tf.sqrt(6 / (9 * (K + L)))))
    B5 = tf.Variable(tf.constant(bias_init, tf.float32, [L]))
    W6 = tf.Variable(tf.random.uniform([5, 5, L, M],
                                       minval=-tf.sqrt(6 / (25 * (L + M))),
                                       maxval=tf.sqrt(6 / (25 * (L + M)))))
    B6 = tf.Variable(tf.constant(bias_init, tf.float32, [M]))
    # W15 = tf.Variable(tf.random.uniform([28 * 28 * J, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 28 * 28 * J)),
    #                                   maxval=tf.sqrt(0.01 / (N + 28 * 28 * J))))
    # W25 = tf.Variable(tf.random.uniform([14 * 14 * K, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * K)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * K))))
    # W35 = tf.Variable(tf.random.uniform([14 * 14 * L, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * L)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * L))))
    W7 = tf.Variable(tf.random.uniform([16 * M, N],
                                       minval=-tf.sqrt(6 / (N + 16 * M)),
                                       maxval=tf.sqrt(6 / (N + 16 * M))))
    B7 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))

    W8 = tf.Variable(tf.random.uniform([N, N2], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B8 = tf.Variable(tf.constant(bias_init, tf.float32, [N2]))
    W9 = tf.Variable(tf.random.uniform([N2, N], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B9 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    # W58 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(0.01 / (N + 10)), maxval=tf.sqrt(0.01 / (N + 10))))
    # W68 = tf.Variable(tf.random.uniform([N2, 10], minval=-tf.sqrt(0.01 / (N2 + 10)), maxval=tf.sqrt(0.01 / (N2 + 10))))

    W10 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(6 / (N + 10)), maxval=tf.sqrt(6 / (N + 10))))
    B10 = tf.Variable(tf.constant(bias_init, tf.float32, [10]))

    W37 = tf.Variable(tf.random.uniform([144 * J, N],
                                        minval=-tf.sqrt(6 / 144 * J + N),
                                        maxval=tf.sqrt(6 / 144 * J + N)))
    B37 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    s37 = tf.Variable(tf.constant(1, tf.float32, [1, J]))



    s = tf.Variable(tf.constant([2, 2], tf.float32, [2]))

    # stride = 1  # output is 28x28
    # Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    # stride = 2  # output is 14x14
    # Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    # stride = 2  # output is 7x7
    # Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    stride = 1  # output is 28x28
    # Y0 = tf.nn.relu(tf.nn.conv2d(X, W0, strides=[1, stride, stride, 1], padding='SAME') + B0)

    Y1a = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + B1
    Y1b, update_ema1 = batchnorm(Y1a, tst, step, B1, convolutional=True)
    Y1 = tf.nn.relu(Y1b)
    #Y1 = tf.nn.dropout(Y1c, pkeep)

    Y2a = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2
    Y2b, update_ema2 = batchnorm(Y2a, tst, step, B2, convolutional=True)
    Y2 = tf.nn.relu(Y2b)
    #Y2 = tf.nn.dropout(Y2c, pkeep)

    Y3a = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3b, update_ema3 = batchnorm(Y3a, tst, step, B3, convolutional=True)
    Y3c = tf.nn.relu(Y3b)
    Y3d = tf.nn.dropout(Y3c, pkeep)
    Y3 = tf.nn.max_pool2d(Y3d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    Y4a = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4
    Y4b, update_ema4 = batchnorm(Y4a, tst, step, B4, convolutional=True)
    Y4 = tf.nn.relu(Y4b)
    #Y4 = tf.nn.dropout(Y4c, pkeep)

    Y5a = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5
    Y5b, update_ema5 = batchnorm(Y5a, tst, step, B5, convolutional=True)
    Y5 = tf.nn.relu(Y5b)
    #Y5 = tf.nn.dropout(Y5c, pkeep)

    Y6a = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6
    Y6b, update_ema6 = batchnorm(Y6a, tst, step, B6, convolutional=True)
    Y6c = tf.nn.relu(Y6b)
    Y6d = tf.nn.dropout(Y6c, pkeep)
    Y6 = tf.nn.max_pool2d(Y6d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    # reshape the output from the third convolution for the fully connected layer
    # YY1 = tf.reshape(Y1, shape=[-1, 24 * 24 * J])
    # YY2 = tf.reshape(Y2, shape=[-1, 12 * 12 * K])
    # YY3 = tf.reshape(Y3, shape=[-1, 12 * 12 * L])
    Y7 = tf.reduce_mean(tf.reshape(Y6, shape=[-1, 16, M]), axis=1)
    #Y7 = tf.reduce_mean(tf.reshape(Y6d, shape=[-1, 64, M]), axis=1)

    # Y5 = tf.matmul(YY1, W15) + tf.matmul(YY2, W25) + tf.matmul(YY3, W35) + tf.matmul(YY, W5) + B5
    #Y7 = tf.matmul(YY, W7) + B7
    Y37 = tf.reduce_mean(tf.reduce_mean(Y3, axis=1), axis=1)#tf.matmul(tf.reshape(Y3, shape=[-1, 144 * J]), W37) + B37



    Y8a = tf.matmul(Y7, W8) + B8
    Y8b, update_ema8 = batchnorm(Y8a, tst, step, B8, convolutional=False)
    Y8 = tf.nn.relu(Y8b)
    #Y8 = tf.nn.dropout(Y8c, pkeep2)

    Y9a = tf.matmul(Y8, W9) + B9
    Y9b, update_ema9 = batchnorm(Y9a, tst, step, B9, convolutional=False)
    Y9c = tf.nn.relu(Y9b)
    Y9 = tf.nn.dropout(Y9c, pkeep2)

    Ylogits = tf.matmul(Y9, W10) + B10

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema8,
                          update_ema9)

    s = tf.maximum(s, 0.3)

    #kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    #weight = s[0] / tf.reduce_mean(kernel0)
    #kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sim_mat = tf.reduce_sum(tf.reshape(Y_ + 0.1, (-1, 1, 10)) * tf.reshape(Y_ + 0.1, (1, -1, 10)), axis=2)
    all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
    same_mat = sim_mat * all_mat
    diff_mat = all_mat * (1 - sim_mat)

    # kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    # weight = s[0] / tf.reduce_mean(kernel0)
    # kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sq_YY0 = tf.reduce_sum(tf.square(Y37), axis=1)
    sq_YY0 = tf.reshape(sq_YY0, (-1, 1))

    pdist0 = -2 * tf.matmul(Y37, tf.transpose(Y37))
    pdist0 = pdist0 + sq_YY0
    pdist0 = pdist0 + tf.transpose(sq_YY0)
    kernel0 = tf.exp(-s[0] * pdist0 / tf.reduce_mean(pdist0))

    sq_YY = tf.reduce_sum(tf.square(Y7), axis=1)
    sq_YY = tf.reshape(sq_YY, (-1, 1))

    pdist = -2 * tf.matmul(Y7, tf.transpose(Y7))
    pdist = pdist + sq_YY
    pdist = pdist + tf.transpose(sq_YY)
    kernel = tf.exp(-s[1] * pdist / tf.reduce_mean(pdist))

    presig = tf.range(tf.shape(Y_)[0])
    # presig = tf.random_shuffle(presig)
    sig = presig[:tf.shape(Y_)[0] // 2]  # Gets the indicies of a random sample of half of the batch/data
    sig1 = presig[tf.shape(Y_)[0] // 2:]

    pi = tf.one_hot(sig, tf.shape(Y_)[0])
    pi1 = tf.one_hot(sig1, tf.shape(Y_)[0])

    smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    smallkernel1 = tf.matmul(pi1, tf.matmul(kernel, tf.linalg.matrix_transpose(pi1)))

    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff = tf.linalg.solve(smallkernel, Y_small)
    coeff1 = tf.linalg.solve(smallkernel1, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
    test_kernel = kernel[:, :tf.shape(Y_)[0] // 2]
    test_kernel1 = kernel[:, tf.shape(Y_)[0] // 2:]

    net_class = tf.matmul(test_kernel, coeff)
    net_class1 = tf.matmul(test_kernel1, coeff1)

    test_class = tf.matmul(pi_full, Y_)

    class_error = tf.reduce_sum(tf.square(net_class - test_class))
    class_error += tf.reduce_sum(tf.square(net_class1 - test_class))

    norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

    ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
    norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

    rho2 = 1 - norm_sig / norm_dagger

    smallkernel0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    smallkernel01 = tf.matmul(pi1, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi1)))
    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff0 = tf.linalg.solve(smallkernel0, Y_small)
    coeff01 = tf.linalg.solve(smallkernel01, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel0 = tf.matmul(pi_full, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi_full)))
    test_kernel0 = kernel0[:, :tf.shape(Y_)[0] // 2]
    test_kernel01 = kernel0[:, tf.shape(Y_)[0] // 2:]

    net_class0 = tf.matmul(test_kernel0, coeff0)
    net_class01 = tf.matmul(test_kernel01, coeff01)
    test_class0 = tf.matmul(pi_full, Y_)

    class_error0 = tf.reduce_sum(tf.square(net_class0 - test_class0)) + tf.reduce_sum(
        tf.square(net_class01 - test_class0))

    norm_dagger0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel0, Y_)))

    ker_bar0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    A_til0 = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar0, pi))
    norm_sig0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til0, Y_)))

    rho20 = 1 - norm_sig0 / norm_dagger0

    same_dist = tf.reduce_sum(pdist * same_mat) / tf.reduce_sum(same_mat)
    diff_dist = tf.reduce_sum(pdist * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist = tf.reduce_sum(pdist * all_mat) / tf.reduce_sum(all_mat)
    same_dist0 = tf.reduce_sum(pdist0 * same_mat) / tf.reduce_sum(same_mat)
    diff_dist0 = tf.reduce_sum(pdist0 * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist0 = tf.reduce_sum(pdist0 * all_mat) / tf.reduce_sum(all_mat)

    dists = [same_dist, diff_dist, all_dist, same_dist0, diff_dist0, all_dist0]


    # mse = tf.reduce_mean(tf.square(Ylogits - Y_))
    mse = tf.reduce_mean(tf.square(Ylogits - (Y_ + 0.1)))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits + 0.1, labels=(Y_ + 0.1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0000001 + tf.train.exponential_decay(0.001, step, 1000, 1 / math.e)  # cut lr in future
    train_step = tf.train.AdamOptimizer(lr).minimize(0.0 * class_error0 + 0.0 * class_error + cross_entropy)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(10 * lr, 0.9).minimize(0.1 * class_error + cross_entropy)
    train_step1 = tf.train.AdamOptimizer(lr).minimize(class_error, var_list=[W1, B1, W2, B2, W3, B3, W4, B4])
    train_step2 = tf.train.AdamOptimizer(lr).minimize(mse, var_list=[W5, B5, W6, B6, W7, B7])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    acc_max = 0
    a = 0.3
    b = 0.3
    r = 0.1
    bs = 100
    err = np.zeros((121, 2, 2))
    rhoce = np.zeros((121, 3))
    distances = np.zeros((121, 6))
    acc = np.zeros((20))
    acct = np.zeros((120))
    for i in range(10001):
        # batch_X, batch_Y = get_batch(i)
        # if i < 5000: batch_X, batch_Y = get_rand_batch3(i, 600, tr_im, tr_ohid)
        # else: batch_X, batch_Y = get_rand_batch3(i, 100, tr_im, tr_ohid)
        # batch_size = int(150 + 2 * math.floor(375 * math.exp(-i / 1000.0)))
        # batch_size = 300

        # batch_X, batch_Y = get_def_rand_batch2(i, batch_size, tr_im, tr_ohid)
        batch_X, batch_Y = get_rand_batch4(bs)
        # for i in range(10):
        #    plt.imshow(batch_X[i].reshape((28, 28)))
        #    plt.show()
        # batch_X, batch_Y = get_rand_batch4(batch_size)

        sess.run(update_ema,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})
        sess.run(train_step,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: pk, pkeep2: pk2, tst: False, ba_size: bs})

        if i % 100 == 0:
            sig, rho1, ce1, ce0, cr1, ds = sess.run([s, rho2, class_error, class_error0, cross_entropy, dists],
                                                    {X0: batch_X, Y_: batch_Y, step: i, tst: False, pkeep: 1.0,
                                                     pkeep2: 1.0,
                                                     ba_size: bs})
            distances[i // 100] = np.asarray(ds)
            rhoce[i // 100, 0] = rho1
            rhoce[i // 100, 1] = ce0
            rhoce[i // 100, 2] = ce1
            print(i, "rho: ", rho1, " class error: ", ce0, ce1, "cross ent:", cr1, "sig:", sig, "dists:", distances[i // 100], time.localtime())
        pkeep_i = 0.75  # + 0.25 * (math.exp(-(i - 0) / 1000))
        # if i < 5000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step2, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # if i < 3000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})

        if i % 1000 == 0 and i >= 0:
            acc = np.zeros((20))
            c = np.zeros((20))
            for j in range(20):
                acc[j], c[j] = sess.run([accuracy, cross_entropy],
                                        {X0: te_im[j * 500: (j + 1) * 500],
                                         Y_: te_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                         ba_size: bs})

            err[i // 100, 0, 0] = 1 - acc.mean()
            err[i // 100, 0, 1] = c.mean()
            # a, c = sess.run([accuracy, cross_entropy], {X0: te_im, Y_id: te_id, pkeep: 1.0})
            if acc.mean() > acc_max:
                acc_max = acc.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* test accuracy:" + str(
                acc.mean()) + " test loss: " + str(c.mean()) + " max accuracy:" + str(acc_max))
        if i == 12000:
            acct = np.zeros((120))
            ct = np.zeros((120))
            for j in range(120):
                acct[j], ct[j] = sess.run([accuracy, cross_entropy],
                                          {X0: tr_im[j * 500: (j + 1) * 500],
                                           Y_: tr_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                           ba_size: bs})
            err[i // 100, 1, 0] = 1 - acct.mean()
            err[i // 100, 1, 1] = ct.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* train accuracy:" + str(
                acct.mean()) + " train loss: " + str(ct.mean()))

    return acc.mean(), acct.mean(), acc_max, err, rhoce, distances

def CNN_KF18kf2_RBF_AUG(lam3=0.01, lam6=0.1, gamma_fac=1.0):
    # Uses all 60K MNIST training points
    # CNN with KF training with rbf kernel
    X0 = tf.placeholder(tf.float32, [None, 28, 28])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    pkeep2 = tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)

    ba_size = tf.placeholder(tf.float32)

    X = tf.reshape(X0, [-1, 28, 28, 1])
    # Y_ = tf.one_hot(Y_id, 10)

    H = 150
    I = 150
    J = 150
    K = 300
    L = 300
    M = 300
    full_layer_size = 300
    N2 = 1200
    N = full_layer_size
    width = N
    in_fac = 1
    bias_init = 0.01

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.9,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    W1 = tf.Variable(tf.random.uniform([3, 3, 1, H],
                                       minval=-tf.sqrt(6 / (9 * (1 + H))),
                                       maxval=tf.sqrt(6 / (9 * (1 + H)))))
    B1 = tf.Variable(tf.constant(bias_init, tf.float32, [H]))
    W2 = tf.Variable(tf.random.uniform([3, 3, H, I],
                                       minval=-tf.sqrt(6 / (9 * (H + I))),
                                       maxval=tf.sqrt(6 / (9 * (H + I)))))
    B2 = tf.Variable(tf.constant(bias_init, tf.float32, [I]))
    W3 = tf.Variable(tf.random.uniform([5, 5, I, J],
                                       minval=-tf.sqrt(6 / (25 * (I + J))),
                                       maxval=tf.sqrt(6 / (25 * (I + J)))))
    B3 = tf.Variable(tf.constant(bias_init, tf.float32, [J]))
    W4 = tf.Variable(tf.random.uniform([3, 3, J, K],
                                       minval=-tf.sqrt(6 / (9 * (J + K))),
                                       maxval=tf.sqrt(6 / (9 * (J + K)))))
    B4 = tf.Variable(tf.constant(bias_init, tf.float32, [K]))
    W5 = tf.Variable(tf.random.uniform([3, 3, K, L],
                                       minval=-tf.sqrt(6 / (9 * (K + L))),
                                       maxval=tf.sqrt(6 / (9 * (K + L)))))
    B5 = tf.Variable(tf.constant(bias_init, tf.float32, [L]))
    W6 = tf.Variable(tf.random.uniform([5, 5, L, M],
                                       minval=-tf.sqrt(6 / (25 * (L + M))),
                                       maxval=tf.sqrt(6 / (25 * (L + M)))))
    B6 = tf.Variable(tf.constant(bias_init, tf.float32, [M]))
    # W15 = tf.Variable(tf.random.uniform([28 * 28 * J, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 28 * 28 * J)),
    #                                   maxval=tf.sqrt(0.01 / (N + 28 * 28 * J))))
    # W25 = tf.Variable(tf.random.uniform([14 * 14 * K, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * K)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * K))))
    # W35 = tf.Variable(tf.random.uniform([14 * 14 * L, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * L)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * L))))
    W7 = tf.Variable(tf.random.uniform([16 * M, N],
                                       minval=-tf.sqrt(6 / (N + 16 * M)),
                                       maxval=tf.sqrt(6 / (N + 16 * M))))
    B7 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))

    W8 = tf.Variable(tf.random.uniform([N, N2], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B8 = tf.Variable(tf.constant(bias_init, tf.float32, [N2]))
    W9 = tf.Variable(tf.random.uniform([N2, N], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B9 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    # W58 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(0.01 / (N + 10)), maxval=tf.sqrt(0.01 / (N + 10))))
    # W68 = tf.Variable(tf.random.uniform([N2, 10], minval=-tf.sqrt(0.01 / (N2 + 10)), maxval=tf.sqrt(0.01 / (N2 + 10))))

    W10 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(6 / (N + 10)), maxval=tf.sqrt(6 / (N + 10))))
    B10 = tf.Variable(tf.constant(bias_init, tf.float32, [10]))

    W37 = tf.Variable(tf.random.uniform([144 * J, N],
                                        minval=-tf.sqrt(6 / 144 * J + N),
                                        maxval=tf.sqrt(6 / 144 * J + N)))
    B37 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    s37 = tf.Variable(tf.constant(1, tf.float32, [1, J]))



    s = tf.Variable(tf.constant([gamma_fac, gamma_fac], tf.float32, [2]))

    # stride = 1  # output is 28x28
    # Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    # stride = 2  # output is 14x14
    # Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    # stride = 2  # output is 7x7
    # Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    stride = 1  # output is 28x28
    # Y0 = tf.nn.relu(tf.nn.conv2d(X, W0, strides=[1, stride, stride, 1], padding='SAME') + B0)

    Y1a = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + B1
    Y1b, update_ema1 = batchnorm(Y1a, tst, step, B1, convolutional=True)
    Y1c = tf.nn.relu(Y1b)
    Y1 = tf.nn.dropout(Y1c, pkeep)

    Y2a = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2
    Y2b, update_ema2 = batchnorm(Y2a, tst, step, B2, convolutional=True)
    Y2c = tf.nn.relu(Y2b)
    Y2 = tf.nn.dropout(Y2c, pkeep)

    Y3a = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3b, update_ema3 = batchnorm(Y3a, tst, step, B3, convolutional=True)
    Y3c = tf.nn.relu(Y3b)
    Y3d = tf.nn.dropout(Y3c, pkeep)
    Y3 = tf.nn.max_pool2d(Y3d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    Y4a = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4
    Y4b, update_ema4 = batchnorm(Y4a, tst, step, B4, convolutional=True)
    Y4c = tf.nn.relu(Y4b)
    Y4 = tf.nn.dropout(Y4c, pkeep)

    Y5a = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5
    Y5b, update_ema5 = batchnorm(Y5a, tst, step, B5, convolutional=True)
    Y5c = tf.nn.relu(Y5b)
    Y5 = tf.nn.dropout(Y5c, pkeep)

    Y6a = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6
    Y6b, update_ema6 = batchnorm(Y6a, tst, step, B6, convolutional=True)
    Y6c = tf.nn.relu(Y6b)
    Y6d = tf.nn.dropout(Y6c, pkeep)
    Y6 = tf.nn.max_pool2d(Y6d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    # reshape the output from the third convolution for the fully connected layer
    # YY1 = tf.reshape(Y1, shape=[-1, 24 * 24 * J])
    # YY2 = tf.reshape(Y2, shape=[-1, 12 * 12 * K])
    # YY3 = tf.reshape(Y3, shape=[-1, 12 * 12 * L])
    Y7 = tf.reduce_mean(tf.reshape(Y6, shape=[-1, 16, M]), axis=1)
    #Y7 = tf.reduce_mean(tf.reshape(Y6d, shape=[-1, 64, M]), axis=1)

    # Y5 = tf.matmul(YY1, W15) + tf.matmul(YY2, W25) + tf.matmul(YY3, W35) + tf.matmul(YY, W5) + B5
    #Y7 = tf.matmul(YY, W7) + B7
    Y37 = tf.reduce_mean(tf.reduce_mean(Y3, axis=1), axis=1)#tf.matmul(tf.reshape(Y3, shape=[-1, 144 * J]), W37) + B37



    Y8a = tf.matmul(Y7, W8) + B8
    Y8b, update_ema8 = batchnorm(Y8a, tst, step, B8, convolutional=False)
    Y8c = tf.nn.relu(Y8b)
    Y8 = tf.nn.dropout(Y8c, pkeep2)

    Y9a = tf.matmul(Y8, W9) + B9
    Y9b, update_ema9 = batchnorm(Y9a, tst, step, B9, convolutional=False)
    Y9c = tf.nn.relu(Y9b)
    Y9 = tf.nn.dropout(Y9c, pkeep2)

    Ylogits = tf.matmul(Y9, W10) + B10

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema8,
                          update_ema9)

    s = tf.maximum(s, 0.3)

    #kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    #weight = s[0] / tf.reduce_mean(kernel0)
    #kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sim_mat = tf.reduce_sum(tf.reshape(Y_ + 0.1, (-1, 1, 10)) * tf.reshape(Y_ + 0.1, (1, -1, 10)), axis=2)
    all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
    same_mat = sim_mat * all_mat
    diff_mat = all_mat * (1 - sim_mat)

    # kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    # weight = s[0] / tf.reduce_mean(kernel0)
    # kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sq_YY0 = tf.reduce_sum(tf.square(Y37), axis=1)
    sq_YY0 = tf.reshape(sq_YY0, (-1, 1))

    pdist0 = -2 * tf.matmul(Y37, tf.transpose(Y37))
    pdist0 = pdist0 + sq_YY0
    pdist0 = pdist0 + tf.transpose(sq_YY0)
    kernel0 = tf.exp(-s[0] * pdist0 / tf.reduce_mean(pdist0))

    sq_YY = tf.reduce_sum(tf.square(Y7), axis=1)
    sq_YY = tf.reshape(sq_YY, (-1, 1))

    pdist = -2 * tf.matmul(Y7, tf.transpose(Y7))
    pdist = pdist + sq_YY
    pdist = pdist + tf.transpose(sq_YY)
    kernel = tf.exp(-s[1] * pdist / tf.reduce_mean(pdist))

    presig = tf.range(tf.shape(Y_)[0])
    # presig = tf.random_shuffle(presig)
    sig = presig[:tf.shape(Y_)[0] // 2]  # Gets the indicies of a random sample of half of the batch/data
    sig1 = presig[tf.shape(Y_)[0] // 2:]

    pi = tf.one_hot(sig, tf.shape(Y_)[0])
    pi1 = tf.one_hot(sig1, tf.shape(Y_)[0])

    smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    smallkernel1 = tf.matmul(pi1, tf.matmul(kernel, tf.linalg.matrix_transpose(pi1)))

    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff = tf.linalg.solve(smallkernel, Y_small)
    coeff1 = tf.linalg.solve(smallkernel1, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
    test_kernel = kernel[:, :tf.shape(Y_)[0] // 2]
    test_kernel1 = kernel[:, tf.shape(Y_)[0] // 2:]

    net_class = tf.matmul(test_kernel, coeff)
    net_class1 = tf.matmul(test_kernel1, coeff1)

    test_class = tf.matmul(pi_full, Y_)

    class_error = tf.reduce_sum(tf.square(net_class - test_class))
    class_error += tf.reduce_sum(tf.square(net_class1 - test_class))

    norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

    ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
    norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

    rho2 = 1 - norm_sig / norm_dagger

    smallkernel0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    smallkernel01 = tf.matmul(pi1, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi1)))
    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff0 = tf.linalg.solve(smallkernel0, Y_small)
    coeff01 = tf.linalg.solve(smallkernel01, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel0 = tf.matmul(pi_full, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi_full)))
    test_kernel0 = kernel0[:, :tf.shape(Y_)[0] // 2]
    test_kernel01 = kernel0[:, tf.shape(Y_)[0] // 2:]

    net_class0 = tf.matmul(test_kernel0, coeff0)
    net_class01 = tf.matmul(test_kernel01, coeff01)
    test_class0 = tf.matmul(pi_full, Y_)

    class_error0 = tf.reduce_sum(tf.square(net_class0 - test_class0)) + tf.reduce_sum(
        tf.square(net_class01 - test_class0))

    norm_dagger0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel0, Y_)))

    ker_bar0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    A_til0 = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar0, pi))
    norm_sig0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til0, Y_)))

    rho20 = 1 - norm_sig0 / norm_dagger0

    same_dist = tf.reduce_sum(pdist * same_mat) / tf.reduce_sum(same_mat)
    diff_dist = tf.reduce_sum(pdist * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist = tf.reduce_sum(pdist * all_mat) / tf.reduce_sum(all_mat)
    same_dist0 = tf.reduce_sum(pdist0 * same_mat) / tf.reduce_sum(same_mat)
    diff_dist0 = tf.reduce_sum(pdist0 * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist0 = tf.reduce_sum(pdist0 * all_mat) / tf.reduce_sum(all_mat)

    dists = [same_dist, diff_dist, all_dist, same_dist0, diff_dist0, all_dist0]

    # mse = tf.reduce_mean(tf.square(Ylogits - Y_))
    mse = tf.reduce_mean(tf.square(Ylogits - (Y_ + 0.1)))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits + 0.1, labels=(Y_ + 0.1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0000001 + tf.train.exponential_decay(0.001, step, 1000, 1 / math.e)  # cut lr in future
    train_step = tf.train.AdamOptimizer(lr).minimize(lam3 * class_error0 + lam6 * class_error + cross_entropy)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(10 * lr, 0.9).minimize(0.1 * class_error + cross_entropy)
    train_step1 = tf.train.AdamOptimizer(lr).minimize(class_error, var_list=[W1, B1, W2, B2, W3, B3, W4, B4])
    train_step2 = tf.train.AdamOptimizer(lr).minimize(mse, var_list=[W5, B5, W6, B6, W7, B7])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    acc_max = 0
    a = 0.3
    b = 0.3
    r = 0.1
    bs = 100
    err = np.zeros((121, 2, 2))
    rhoce = np.zeros((121, 3))
    distances = np.zeros((121, 6))
    acc = np.zeros((20))
    acct = np.zeros((120))
    for i in range(10001):
        # batch_X, batch_Y = get_batch(i)
        # if i < 5000: batch_X, batch_Y = get_rand_batch3(i, 600, tr_im, tr_ohid)
        # else: batch_X, batch_Y = get_rand_batch3(i, 100, tr_im, tr_ohid)
        # batch_size = int(150 + 2 * math.floor(375 * math.exp(-i / 1000.0)))
        # batch_size = 300

        # batch_X, batch_Y = get_def_rand_batch2(i, batch_size, tr_im, tr_ohid)
        batch_X, batch_Y = get_rand_batch4(bs)
        # for i in range(10):
        #    plt.imshow(batch_X[i].reshape((28, 28)))
        #    plt.show()
        # batch_X, batch_Y = get_rand_batch4(batch_size)

        sess.run(update_ema,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})
        sess.run(train_step,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})

        if i % 100 == 0:
            sig, rho1, ce1, ce0, cr1, ds = sess.run([s, rho2, class_error, class_error0, cross_entropy, dists],
                                                {X0: batch_X, Y_: batch_Y, step: i, tst: False, pkeep: 1.0, pkeep2: 1.0,
                                                 ba_size: bs})
            distances[i // 100] = np.asarray(ds)
            rhoce[i // 100, 0] = rho1
            rhoce[i // 100, 1] = ce0
            rhoce[i // 100, 2] = ce1
            print(i, "rho: ", rho1, " class error: ", ce0, ce1, "cross ent:", cr1, "sig:", sig, "dists:", distances[i // 100], time.localtime())
        pkeep_i = 0.75  # + 0.25 * (math.exp(-(i - 0) / 1000))
        # if i < 5000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step2, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # if i < 3000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})

        if i % 1000 == 0 and i >= 0:
            acc = np.zeros((20))
            c = np.zeros((20))
            for j in range(20):
                acc[j], c[j] = sess.run([accuracy, cross_entropy],
                                        {X0: te_im[j * 500: (j + 1) * 500],
                                         Y_: te_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                         ba_size: bs})

            err[i // 100, 0, 0] = 1 - acc.mean()
            err[i // 100, 0, 1] = c.mean()
            # a, c = sess.run([accuracy, cross_entropy], {X0: te_im, Y_id: te_id, pkeep: 1.0})
            if acc.mean() > acc_max:
                acc_max = acc.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* test accuracy:" + str(
                acc.mean()) + " test loss: " + str(c.mean()) + " max accuracy:" + str(acc_max))

        if i == 12000:
            acct = np.zeros((120))
            ct = np.zeros((120))
            for j in range(120):
                acct[j], ct[j] = sess.run([accuracy, cross_entropy],
                                          {X0: tr_im[j * 500: (j + 1) * 500],
                                           Y_: tr_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                           ba_size: bs})
            err[i // 100, 1, 0] = 1 - acct.mean()
            err[i // 100, 1, 1] = ct.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* train accuracy:" + str(
                acct.mean()) + " train loss: " + str(ct.mean()))

    return acc.mean(), acct.mean(), acc_max, err, rhoce, distances

def CNN_KF180bn_RBF_AUG():
    # Uses all 60K MNIST training points
    # CNN with KF training with rbf kernel
    #tf.random.set_random_seed(0)
    #np.random.seed(0)
    X0 = tf.placeholder(tf.float32, [None, 28, 28])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    pkeep2 = tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)

    ba_size = tf.placeholder(tf.float32)

    X = tf.reshape(X0, [-1, 28, 28, 1])
    # Y_ = tf.one_hot(Y_id, 10)

    H = 150
    I = 150
    J = 150
    K = 300
    L = 300
    M = 300
    full_layer_size = 300
    N2 = 1200
    N = full_layer_size
    width = N
    in_fac = 1
    bias_init = 0.01

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.9,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    W1 = tf.Variable(tf.random.uniform([3, 3, 1, H],
                                       minval=-tf.sqrt(6. / (9 * (1 + H))),
                                       maxval=tf.sqrt(6. / (9 * (1 + H)))))
    B1 = tf.Variable(tf.constant(bias_init, tf.float32, [H]))
    W2 = tf.Variable(tf.random.uniform([3, 3, H, I],
                                       minval=-tf.sqrt(6. / (9 * (H + I))),
                                       maxval=tf.sqrt(6. / (9 * (H + I)))))
    B2 = tf.Variable(tf.constant(bias_init, tf.float32, [I]))
    W3 = tf.Variable(tf.random.uniform([5, 5, I, J],
                                       minval=-tf.sqrt(6. / (25 * (I + J))),
                                       maxval=tf.sqrt(6. / (25 * (I + J)))))
    B3 = tf.Variable(tf.constant(bias_init, tf.float32, [J]))
    W4 = tf.Variable(tf.random.uniform([3, 3, J, K],
                                       minval=-tf.sqrt(6. / (9 * (J + K))),
                                       maxval=tf.sqrt(6. / (9 * (J + K)))))
    B4 = tf.Variable(tf.constant(bias_init, tf.float32, [K]))
    W5 = tf.Variable(tf.random.uniform([3, 3, K, L],
                                       minval=-tf.sqrt(6. / (9 * (K + L))),
                                       maxval=tf.sqrt(6. / (9 * (K + L)))))
    B5 = tf.Variable(tf.constant(bias_init, tf.float32, [L]))
    W6 = tf.Variable(tf.random.uniform([5, 5, L, M],
                                       minval=-tf.sqrt(6. / (25 * (L + M))),
                                       maxval=tf.sqrt(6. / (25 * (L + M)))))
    B6 = tf.Variable(tf.constant(bias_init, tf.float32, [M]))
    # W15 = tf.Variable(tf.random.uniform([28 * 28 * J, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 28 * 28 * J)),
    #                                   maxval=tf.sqrt(0.01 / (N + 28 * 28 * J))))
    # W25 = tf.Variable(tf.random.uniform([14 * 14 * K, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * K)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * K))))
    # W35 = tf.Variable(tf.random.uniform([14 * 14 * L, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * L)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * L))))
    W7 = tf.Variable(tf.random.uniform([16 * M, N],
                                       minval=-tf.sqrt(6. / (N + 16 * M)),
                                       maxval=tf.sqrt(6. / (N + 16 * M))))
    B7 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))

    W8 = tf.Variable(tf.random.uniform([N, N2], minval=-tf.sqrt(6. / (N + N2)), maxval=tf.sqrt(6. / (N + N2))))
    B8 = tf.Variable(tf.constant(bias_init, tf.float32, [N2]))
    W9 = tf.Variable(tf.random.uniform([N2, N], minval=-tf.sqrt(6. / (N + N2)), maxval=tf.sqrt(6. / (N + N2))))
    B9 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    # W58 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(0.01 / (N + 10)), maxval=tf.sqrt(0.01 / (N + 10))))
    # W68 = tf.Variable(tf.random.uniform([N2, 10], minval=-tf.sqrt(0.01 / (N2 + 10)), maxval=tf.sqrt(0.01 / (N2 + 10))))

    W10 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(6. / (N + 10)), maxval=tf.sqrt(6. / (N + 10))))
    B10 = tf.Variable(tf.constant(bias_init, tf.float32, [10]))

    W37 = tf.Variable(tf.random.uniform([144 * J, N],
                                        minval=-tf.sqrt(6. / 144 * J + N),
                                        maxval=tf.sqrt(6. / 144 * J + N)))
    B37 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    s37 = tf.Variable(tf.constant(1, tf.float32, [1, J]))



    s = tf.Variable(tf.constant([2, 2], tf.float32, [2]))

    # stride = 1  # output is 28x28
    # Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    # stride = 2  # output is 14x14
    # Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    # stride = 2  # output is 7x7
    # Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    stride = 1  # output is 28x28
    # Y0 = tf.nn.relu(tf.nn.conv2d(X, W0, strides=[1, stride, stride, 1], padding='SAME') + B0)

    Y1a = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + B1
    Y1b, update_ema1 = batchnorm(Y1a, tst, step, B1, convolutional=True)
    Y1c = tf.nn.relu(Y1b)
    Y1 = tf.nn.dropout(Y1c, pkeep)

    Y2a = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2
    Y2b, update_ema2 = batchnorm(Y2a, tst, step, B2, convolutional=True)
    Y2c = tf.nn.relu(Y2b)
    Y2 = tf.nn.dropout(Y2c, pkeep)

    Y3a = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3b, update_ema3 = batchnorm(Y3a, tst, step, B3, convolutional=True)
    Y3c = tf.nn.relu(Y3b)
    Y3d = tf.nn.dropout(Y3c, pkeep)
    Y3 = tf.nn.max_pool2d(Y3d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    Y4a = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4
    Y4b, update_ema4 = batchnorm(Y4a, tst, step, B4, convolutional=True)
    Y4c = tf.nn.relu(Y4b)
    Y4 = tf.nn.dropout(Y4c, pkeep)

    Y5a = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5
    Y5b, update_ema5 = batchnorm(Y5a, tst, step, B5, convolutional=True)
    Y5c = tf.nn.relu(Y5b)
    Y5 = tf.nn.dropout(Y5c, pkeep)

    Y6a = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6
    Y6b, update_ema6 = batchnorm(Y6a, tst, step, B6, convolutional=True)
    Y6c = tf.nn.relu(Y6b)
    Y6d = tf.nn.dropout(Y6c, pkeep)
    Y6 = tf.nn.max_pool2d(Y6d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    # reshape the output from the third convolution for the fully connected layer
    # YY1 = tf.reshape(Y1, shape=[-1, 24 * 24 * J])
    # YY2 = tf.reshape(Y2, shape=[-1, 12 * 12 * K])
    # YY3 = tf.reshape(Y3, shape=[-1, 12 * 12 * L])
    Y7 = tf.reduce_mean(tf.reshape(Y6, shape=[-1, 16, M]), axis=1)
    #Y7 = tf.reduce_mean(tf.reshape(Y6d, shape=[-1, 64, M]), axis=1)

    # Y5 = tf.matmul(YY1, W15) + tf.matmul(YY2, W25) + tf.matmul(YY3, W35) + tf.matmul(YY, W5) + B5
    #Y7 = tf.matmul(YY, W7) + B7
    Y37 = tf.reduce_mean(tf.reduce_mean(Y3, axis=1), axis=1)#tf.matmul(tf.reshape(Y3, shape=[-1, 144 * J]), W37) + B37



    Y8a = tf.matmul(Y7, W8) + B8
    Y8b, update_ema8 = batchnorm(Y8a, tst, step, B8, convolutional=False)
    Y8c = tf.nn.relu(Y8b)
    Y8 = tf.nn.dropout(Y8c, pkeep2)

    Y9a = tf.matmul(Y8, W9) + B9
    Y9b, update_ema9 = batchnorm(Y9a, tst, step, B9, convolutional=False)
    Y9c = tf.nn.relu(Y9b)
    Y9 = tf.nn.dropout(Y9c, pkeep2)

    Ylogits = tf.matmul(Y9, W10) + B10

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema8,
                          update_ema9)

    s = tf.maximum(s, 0.3)

    #kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    #weight = s[0] / tf.reduce_mean(kernel0)
    #kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sim_mat = tf.reduce_sum(tf.reshape(Y_ + 0.1, (-1, 1, 10)) * tf.reshape(Y_ + 0.1, (1, -1, 10)), axis=2)
    all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
    same_mat = sim_mat * all_mat
    diff_mat = all_mat * (1 - sim_mat)

    # kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    # weight = s[0] / tf.reduce_mean(kernel0)
    # kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sq_YY0 = tf.reduce_sum(tf.square(Y37), axis=1)
    sq_YY0 = tf.reshape(sq_YY0, (-1, 1))

    pdist0 = -2 * tf.matmul(Y37, tf.transpose(Y37))
    pdist0 = pdist0 + sq_YY0
    pdist0 = pdist0 + tf.transpose(sq_YY0)
    kernel0 = tf.exp(-s[0] * pdist0 / tf.reduce_mean(pdist0))

    sq_YY = tf.reduce_sum(tf.square(Y7), axis=1)
    sq_YY = tf.reshape(sq_YY, (-1, 1))

    pdist = -2 * tf.matmul(Y7, tf.transpose(Y7))
    pdist = pdist + sq_YY
    pdist = pdist + tf.transpose(sq_YY)
    kernel = tf.exp(-s[1] * pdist / tf.reduce_mean(pdist))

    presig = tf.range(tf.shape(Y_)[0])
    # presig = tf.random_shuffle(presig)
    sig = presig[:tf.shape(Y_)[0] // 2]  # Gets the indicies of a random sample of half of the batch/data
    sig1 = presig[tf.shape(Y_)[0] // 2:]

    pi = tf.one_hot(sig, tf.shape(Y_)[0])
    pi1 = tf.one_hot(sig1, tf.shape(Y_)[0])

    smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    smallkernel1 = tf.matmul(pi1, tf.matmul(kernel, tf.linalg.matrix_transpose(pi1)))

    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff = tf.linalg.solve(smallkernel, Y_small)
    coeff1 = tf.linalg.solve(smallkernel1, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
    test_kernel = kernel[:, :tf.shape(Y_)[0] // 2]
    test_kernel1 = kernel[:, tf.shape(Y_)[0] // 2:]

    net_class = tf.matmul(test_kernel, coeff)
    net_class1 = tf.matmul(test_kernel1, coeff1)

    test_class = tf.matmul(pi_full, Y_)

    class_error = tf.reduce_sum(tf.square(net_class - test_class))
    class_error += tf.reduce_sum(tf.square(net_class1 - test_class))

    norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

    ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
    norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

    rho2 = 1 - norm_sig / norm_dagger

    smallkernel0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    smallkernel01 = tf.matmul(pi1, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi1)))
    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff0 = tf.linalg.solve(smallkernel0, Y_small)
    coeff01 = tf.linalg.solve(smallkernel01, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel0 = tf.matmul(pi_full, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi_full)))
    test_kernel0 = kernel0[:, :tf.shape(Y_)[0] // 2]
    test_kernel01 = kernel0[:, tf.shape(Y_)[0] // 2:]

    net_class0 = tf.matmul(test_kernel0, coeff0)
    net_class01 = tf.matmul(test_kernel01, coeff01)
    test_class0 = tf.matmul(pi_full, Y_)

    class_error0 = tf.reduce_sum(tf.square(net_class0 - test_class0)) + tf.reduce_sum(
        tf.square(net_class01 - test_class0))

    norm_dagger0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel0, Y_)))

    ker_bar0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    A_til0 = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar0, pi))
    norm_sig0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til0, Y_)))

    rho20 = 1 - norm_sig0 / norm_dagger0

    same_dist = tf.reduce_sum(pdist * same_mat) / tf.reduce_sum(same_mat)
    diff_dist = tf.reduce_sum(pdist * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist = tf.reduce_sum(pdist * all_mat) / tf.reduce_sum(all_mat)
    same_dist0 = tf.reduce_sum(pdist0 * same_mat) / tf.reduce_sum(same_mat)
    diff_dist0 = tf.reduce_sum(pdist0 * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist0 = tf.reduce_sum(pdist0 * all_mat) / tf.reduce_sum(all_mat)

    dists = [same_dist, diff_dist, all_dist, same_dist0, diff_dist0, all_dist0]


    # mse = tf.reduce_mean(tf.square(Ylogits - Y_))
    mse = tf.reduce_mean(tf.square(Ylogits - (Y_ + 0.1)))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits + 0.1, labels=(Y_ + 0.1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0000001 + tf.train.exponential_decay(0.001, step, 1000, 1 / math.e)  # cut lr in future
    train_step = tf.train.AdamOptimizer(lr).minimize(0.0 * class_error0 + 0.0 * class_error + cross_entropy)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(10 * lr, 0.9).minimize(0.1 * class_error + cross_entropy)
    train_step1 = tf.train.AdamOptimizer(lr).minimize(class_error, var_list=[W1, B1, W2, B2, W3, B3, W4, B4])
    train_step2 = tf.train.AdamOptimizer(lr).minimize(mse, var_list=[W5, B5, W6, B6, W7, B7])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    acc_max = 0
    a = 0.3
    b = 0.3
    r = 0.1
    bs = 100
    err = np.zeros((121, 2, 2))
    rhoce = np.zeros((121, 3))
    distances = np.zeros((121, 6))
    for i in range(10001):
        # batch_X, batch_Y = get_batch(i)
        # if i < 5000: batch_X, batch_Y = get_rand_batch3(i, 600, tr_im, tr_ohid)
        # else: batch_X, batch_Y = get_rand_batch3(i, 100, tr_im, tr_ohid)
        # batch_size = int(150 + 2 * math.floor(375 * math.exp(-i / 1000.0)))
        # batch_size = 300

        # batch_X, batch_Y = get_def_rand_batch2(i, batch_size, tr_im, tr_ohid)
        batch_X, batch_Y = get_rand_batch40(bs)
        # for i in range(10):
        #    plt.imshow(batch_X[i].reshape((28, 28)))
        #    plt.show()
        # batch_X, batch_Y = get_rand_batch4(batch_size)

        sess.run(update_ema,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})
        sess.run(train_step,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})

        if i % 100 == 0:
            sig, rho1, ce1, ce0, cr1, ds = sess.run([s, rho2, class_error, class_error0, cross_entropy, dists],
                                                    {X0: batch_X, Y_: batch_Y, step: i, tst: False, pkeep: 1.0,
                                                     pkeep2: 1.0,
                                                     ba_size: bs})
            distances[i // 100] = np.asarray(ds)
            rhoce[i // 100, 0] = rho1
            rhoce[i // 100, 1] = ce0
            rhoce[i // 100, 2] = ce1
            print(i, "rho: ", rho1, " class error: ", ce0, ce1, "cross ent:", cr1, "sig:", sig, "dists:",
                  distances[i // 100], time.localtime())
        pkeep_i = 0.75  # + 0.25 * (math.exp(-(i - 0) / 1000))
        # if i < 5000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step2, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # if i < 3000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})

        if i % 1000 == 0 and i >= 10000:
            acc = np.zeros((20))
            c = np.zeros((20))
            for j in range(20):
                acc[j], c[j] = sess.run([accuracy, cross_entropy],
                                        {X0: te_im[j * 500: (j + 1) * 500],
                                         Y_: te_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                         ba_size: bs})

            err[i // 100, 0, 0] = 1 - acc.mean()
            err[i // 100, 0, 1] = c.mean()
            # a, c = sess.run([accuracy, cross_entropy], {X0: te_im, Y_id: te_id, pkeep: 1.0})
            if acc.mean() > acc_max:
                acc_max = acc.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* test accuracy:" + str(
                acc.mean()) + " test loss: " + str(c.mean()) + " max accuracy:" + str(acc_max))
        if i == 12000:
            acct = np.zeros((120))
            ct = np.zeros((120))
            for j in range(120):
                acct[j], ct[j] = sess.run([accuracy, cross_entropy],
                                          {X0: tr_im[j * 500: (j + 1) * 500],
                                           Y_: tr_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                           ba_size: bs})
            err[i // 100, 1, 0] = 1 - acct.mean()
            err[i // 100, 1, 1] = ct.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* train accuracy:" + str(
                acct.mean()) + " train loss: " + str(ct.mean()))

    return acc.mean(), acct.mean(), acc_max, err, rhoce, distances

def CNN_KF180do_RBF_AUG():
    # Uses all 60K MNIST training points
    # CNN with KF training with rbf kernel
    X0 = tf.placeholder(tf.float32, [None, 28, 28])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    pkeep2 = tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)

    ba_size = tf.placeholder(tf.float32)

    X = tf.reshape(X0, [-1, 28, 28, 1])
    # Y_ = tf.one_hot(Y_id, 10)

    H = 150
    I = 150
    J = 150
    K = 300
    L = 300
    M = 300
    full_layer_size = 300
    N2 = 1200
    N = full_layer_size
    width = N
    in_fac = 1
    bias_init = 0.01

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.9,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    W1 = tf.Variable(tf.random.uniform([3, 3, 1, H],
                                       minval=-tf.sqrt(6 / (9 * (1 + H))),
                                       maxval=tf.sqrt(6 / (9 * (1 + H)))))
    B1 = tf.Variable(tf.constant(bias_init, tf.float32, [H]))
    W2 = tf.Variable(tf.random.uniform([3, 3, H, I],
                                       minval=-tf.sqrt(6 / (9 * (H + I))),
                                       maxval=tf.sqrt(6 / (9 * (H + I)))))
    B2 = tf.Variable(tf.constant(bias_init, tf.float32, [I]))
    W3 = tf.Variable(tf.random.uniform([5, 5, I, J],
                                       minval=-tf.sqrt(6 / (25 * (I + J))),
                                       maxval=tf.sqrt(6 / (25 * (I + J)))))
    B3 = tf.Variable(tf.constant(bias_init, tf.float32, [J]))
    W4 = tf.Variable(tf.random.uniform([3, 3, J, K],
                                       minval=-tf.sqrt(6 / (9 * (J + K))),
                                       maxval=tf.sqrt(6 / (9 * (J + K)))))
    B4 = tf.Variable(tf.constant(bias_init, tf.float32, [K]))
    W5 = tf.Variable(tf.random.uniform([3, 3, K, L],
                                       minval=-tf.sqrt(6 / (9 * (K + L))),
                                       maxval=tf.sqrt(6 / (9 * (K + L)))))
    B5 = tf.Variable(tf.constant(bias_init, tf.float32, [L]))
    W6 = tf.Variable(tf.random.uniform([5, 5, L, M],
                                       minval=-tf.sqrt(6 / (25 * (L + M))),
                                       maxval=tf.sqrt(6 / (25 * (L + M)))))
    B6 = tf.Variable(tf.constant(bias_init, tf.float32, [M]))
    # W15 = tf.Variable(tf.random.uniform([28 * 28 * J, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 28 * 28 * J)),
    #                                   maxval=tf.sqrt(0.01 / (N + 28 * 28 * J))))
    # W25 = tf.Variable(tf.random.uniform([14 * 14 * K, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * K)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * K))))
    # W35 = tf.Variable(tf.random.uniform([14 * 14 * L, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * L)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * L))))
    W7 = tf.Variable(tf.random.uniform([16 * M, N],
                                       minval=-tf.sqrt(6 / (N + 16 * M)),
                                       maxval=tf.sqrt(6 / (N + 16 * M))))
    B7 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))

    W8 = tf.Variable(tf.random.uniform([N, N2], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B8 = tf.Variable(tf.constant(bias_init, tf.float32, [N2]))
    W9 = tf.Variable(tf.random.uniform([N2, N], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B9 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    # W58 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(0.01 / (N + 10)), maxval=tf.sqrt(0.01 / (N + 10))))
    # W68 = tf.Variable(tf.random.uniform([N2, 10], minval=-tf.sqrt(0.01 / (N2 + 10)), maxval=tf.sqrt(0.01 / (N2 + 10))))

    W10 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(6 / (N + 10)), maxval=tf.sqrt(6 / (N + 10))))
    B10 = tf.Variable(tf.constant(bias_init, tf.float32, [10]))

    W37 = tf.Variable(tf.random.uniform([144 * J, N],
                                        minval=-tf.sqrt(6 / 144 * J + N),
                                        maxval=tf.sqrt(6 / 144 * J + N)))
    B37 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    s37 = tf.Variable(tf.constant(1, tf.float32, [1, J]))



    s = tf.Variable(tf.constant([2, 2], tf.float32, [2]))

    # stride = 1  # output is 28x28
    # Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    # stride = 2  # output is 14x14
    # Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    # stride = 2  # output is 7x7
    # Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    stride = 1  # output is 28x28
    # Y0 = tf.nn.relu(tf.nn.conv2d(X, W0, strides=[1, stride, stride, 1], padding='SAME') + B0)

    Y1a = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + B1
    Y1b, update_ema1 = batchnorm(Y1a, tst, step, B1, convolutional=True)
    Y1c = tf.nn.relu(Y1b)
    Y1 = tf.nn.dropout(Y1c, pkeep)

    Y2a = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2
    Y2b, update_ema2 = batchnorm(Y2a, tst, step, B2, convolutional=True)
    Y2c = tf.nn.relu(Y2b)
    Y2 = tf.nn.dropout(Y2c, pkeep)

    Y3a = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3b, update_ema3 = batchnorm(Y3a, tst, step, B3, convolutional=True)
    Y3c = tf.nn.relu(Y3b)
    Y3d = tf.nn.dropout(Y3c, pkeep)
    Y3 = tf.nn.max_pool2d(Y3d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    Y4a = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4
    Y4b, update_ema4 = batchnorm(Y4a, tst, step, B4, convolutional=True)
    Y4c = tf.nn.relu(Y4b)
    Y4 = tf.nn.dropout(Y4c, pkeep)

    Y5a = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5
    Y5b, update_ema5 = batchnorm(Y5a, tst, step, B5, convolutional=True)
    Y5c = tf.nn.relu(Y5b)
    Y5 = tf.nn.dropout(Y5c, pkeep)

    Y6a = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6
    Y6b, update_ema6 = batchnorm(Y6a, tst, step, B6, convolutional=True)
    Y6c = tf.nn.relu(Y6b)
    Y6d = tf.nn.dropout(Y6c, pkeep)
    Y6 = tf.nn.max_pool2d(Y6d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    # reshape the output from the third convolution for the fully connected layer
    # YY1 = tf.reshape(Y1, shape=[-1, 24 * 24 * J])
    # YY2 = tf.reshape(Y2, shape=[-1, 12 * 12 * K])
    # YY3 = tf.reshape(Y3, shape=[-1, 12 * 12 * L])
    Y7 = tf.reduce_mean(tf.reshape(Y6, shape=[-1, 16, M]), axis=1)
    #Y7 = tf.reduce_mean(tf.reshape(Y6d, shape=[-1, 64, M]), axis=1)

    # Y5 = tf.matmul(YY1, W15) + tf.matmul(YY2, W25) + tf.matmul(YY3, W35) + tf.matmul(YY, W5) + B5
    #Y7 = tf.matmul(YY, W7) + B7
    Y37 = tf.reduce_mean(tf.reduce_mean(Y3, axis=1), axis=1)#tf.matmul(tf.reshape(Y3, shape=[-1, 144 * J]), W37) + B37



    Y8a = tf.matmul(Y7, W8) + B8
    Y8b, update_ema8 = batchnorm(Y8a, tst, step, B8, convolutional=False)
    Y8c = tf.nn.relu(Y8b)
    Y8 = tf.nn.dropout(Y8c, pkeep2)

    Y9a = tf.matmul(Y8, W9) + B9
    Y9b, update_ema9 = batchnorm(Y9a, tst, step, B9, convolutional=False)
    Y9c = tf.nn.relu(Y9b)
    Y9 = tf.nn.dropout(Y9c, pkeep2)

    Ylogits = tf.matmul(Y9, W10) + B10

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema8,
                          update_ema9)

    s = tf.maximum(s, 0.3)

    #kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    #weight = s[0] / tf.reduce_mean(kernel0)
    #kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sim_mat = tf.reduce_sum(tf.reshape(Y_ + 0.1, (-1, 1, 10)) * tf.reshape(Y_ + 0.1, (1, -1, 10)), axis=2)
    all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
    same_mat = sim_mat * all_mat
    diff_mat = all_mat * (1 - sim_mat)

    # kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    # weight = s[0] / tf.reduce_mean(kernel0)
    # kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sq_YY0 = tf.reduce_sum(tf.square(Y37), axis=1)
    sq_YY0 = tf.reshape(sq_YY0, (-1, 1))

    pdist0 = -2 * tf.matmul(Y37, tf.transpose(Y37))
    pdist0 = pdist0 + sq_YY0
    pdist0 = pdist0 + tf.transpose(sq_YY0)
    kernel0 = tf.exp(-s[0] * pdist0 / tf.reduce_mean(pdist0))

    sq_YY = tf.reduce_sum(tf.square(Y7), axis=1)
    sq_YY = tf.reshape(sq_YY, (-1, 1))

    pdist = -2 * tf.matmul(Y7, tf.transpose(Y7))
    pdist = pdist + sq_YY
    pdist = pdist + tf.transpose(sq_YY)
    kernel = tf.exp(-s[1] * pdist / tf.reduce_mean(pdist))

    presig = tf.range(tf.shape(Y_)[0])
    # presig = tf.random_shuffle(presig)
    sig = presig[:tf.shape(Y_)[0] // 2]  # Gets the indicies of a random sample of half of the batch/data
    sig1 = presig[tf.shape(Y_)[0] // 2:]

    pi = tf.one_hot(sig, tf.shape(Y_)[0])
    pi1 = tf.one_hot(sig1, tf.shape(Y_)[0])

    smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    smallkernel1 = tf.matmul(pi1, tf.matmul(kernel, tf.linalg.matrix_transpose(pi1)))

    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff = tf.linalg.solve(smallkernel, Y_small)
    coeff1 = tf.linalg.solve(smallkernel1, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
    test_kernel = kernel[:, :tf.shape(Y_)[0] // 2]
    test_kernel1 = kernel[:, tf.shape(Y_)[0] // 2:]

    net_class = tf.matmul(test_kernel, coeff)
    net_class1 = tf.matmul(test_kernel1, coeff1)

    test_class = tf.matmul(pi_full, Y_)

    class_error = tf.reduce_sum(tf.square(net_class - test_class))
    class_error += tf.reduce_sum(tf.square(net_class1 - test_class))

    norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

    ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
    norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

    rho2 = 1 - norm_sig / norm_dagger

    smallkernel0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    smallkernel01 = tf.matmul(pi1, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi1)))
    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff0 = tf.linalg.solve(smallkernel0, Y_small)
    coeff01 = tf.linalg.solve(smallkernel01, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel0 = tf.matmul(pi_full, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi_full)))
    test_kernel0 = kernel0[:, :tf.shape(Y_)[0] // 2]
    test_kernel01 = kernel0[:, tf.shape(Y_)[0] // 2:]

    net_class0 = tf.matmul(test_kernel0, coeff0)
    net_class01 = tf.matmul(test_kernel01, coeff01)
    test_class0 = tf.matmul(pi_full, Y_)

    class_error0 = tf.reduce_sum(tf.square(net_class0 - test_class0)) + tf.reduce_sum(
        tf.square(net_class01 - test_class0))

    norm_dagger0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel0, Y_)))

    ker_bar0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    A_til0 = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar0, pi))
    norm_sig0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til0, Y_)))

    rho20 = 1 - norm_sig0 / norm_dagger0

    same_dist = tf.reduce_sum(pdist * same_mat) / tf.reduce_sum(same_mat)
    diff_dist = tf.reduce_sum(pdist * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist = tf.reduce_sum(pdist * all_mat) / tf.reduce_sum(all_mat)
    same_dist0 = tf.reduce_sum(pdist0 * same_mat) / tf.reduce_sum(same_mat)
    diff_dist0 = tf.reduce_sum(pdist0 * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist0 = tf.reduce_sum(pdist0 * all_mat) / tf.reduce_sum(all_mat)

    dists = [same_dist, diff_dist, all_dist, same_dist0, diff_dist0, all_dist0]


    # mse = tf.reduce_mean(tf.square(Ylogits - Y_))
    mse = tf.reduce_mean(tf.square(Ylogits - (Y_ + 0.1)))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits + 0.1, labels=(Y_ + 0.1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0000001 + tf.train.exponential_decay(0.001, step, 1000, 1 / math.e)  # cut lr in future
    train_step = tf.train.AdamOptimizer(lr).minimize(0.0 * class_error0 + 0.0 * class_error + cross_entropy)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(10 * lr, 0.9).minimize(0.1 * class_error + cross_entropy)
    train_step1 = tf.train.AdamOptimizer(lr).minimize(class_error, var_list=[W1, B1, W2, B2, W3, B3, W4, B4])
    train_step2 = tf.train.AdamOptimizer(lr).minimize(mse, var_list=[W5, B5, W6, B6, W7, B7])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    acc_max = 0
    a = 0.3
    b = 0.3
    r = 0.1
    bs = 100
    err = np.zeros((121, 2, 2))
    rhoce = np.zeros((121, 3))
    distances = np.zeros((121, 6))
    for i in range(10001):
        # batch_X, batch_Y = get_batch(i)
        # if i < 5000: batch_X, batch_Y = get_rand_batch3(i, 600, tr_im, tr_ohid)
        # else: batch_X, batch_Y = get_rand_batch3(i, 100, tr_im, tr_ohid)
        # batch_size = int(150 + 2 * math.floor(375 * math.exp(-i / 1000.0)))
        # batch_size = 300

        # batch_X, batch_Y = get_def_rand_batch2(i, batch_size, tr_im, tr_ohid)
        batch_X, batch_Y = get_rand_batch40(bs)
        # for i in range(10):
        #    plt.imshow(batch_X[i].reshape((28, 28)))
        #    plt.show()
        # batch_X, batch_Y = get_rand_batch4(batch_size)

        sess.run(update_ema,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})
        sess.run(train_step,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 0.75, pkeep2: 0.75, tst: False, ba_size: bs})

        if i % 100 == 0:
            sig, rho1, ce1, ce0, cr1, ds = sess.run([s, rho2, class_error, class_error0, cross_entropy, dists],
                                                    {X0: batch_X, Y_: batch_Y, step: i, tst: False, pkeep: 1.0,
                                                     pkeep2: 1.0,
                                                     ba_size: bs})
            distances[i // 100] = np.asarray(ds)
            rhoce[i // 100, 0] = rho1
            rhoce[i // 100, 1] = ce0
            rhoce[i // 100, 2] = ce1
            print(i, "rho: ", rho1, " class error: ", ce0, ce1, "cross ent:", cr1, "sig:", sig, "dists:", distances[i // 100], time.localtime())
        pkeep_i = 0.75  # + 0.25 * (math.exp(-(i - 0) / 1000))
        # if i < 5000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step2, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # if i < 3000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})

        if i % 1000 == 0 and i >= 0:
            acc = np.zeros((20))
            c = np.zeros((20))
            for j in range(20):
                acc[j], c[j] = sess.run([accuracy, cross_entropy],
                                        {X0: te_im[j * 500: (j + 1) * 500],
                                         Y_: te_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                         ba_size: bs})

            err[i // 100, 0, 0] = 1 - acc.mean()
            err[i // 100, 0, 1] = c.mean()
            # a, c = sess.run([accuracy, cross_entropy], {X0: te_im, Y_id: te_id, pkeep: 1.0})
            if acc.mean() > acc_max:
                acc_max = acc.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* test accuracy:" + str(
                acc.mean()) + " test loss: " + str(c.mean()) + " max accuracy:" + str(acc_max))
        if i == 12000:
            acct = np.zeros((120))
            ct = np.zeros((120))
            for j in range(120):
                acct[j], ct[j] = sess.run([accuracy, cross_entropy],
                                          {X0: tr_im[j * 500: (j + 1) * 500],
                                           Y_: tr_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                           ba_size: bs})
            err[i // 100, 1, 0] = 1 - acct.mean()
            err[i // 100, 1, 1] = ct.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* train accuracy:" + str(
                acct.mean()) + " train loss: " + str(ct.mean()))

    return acc.mean(), acct.mean(), acc_max, err, rhoce, distances

def CNN_KF180kf2_RBF_AUG():
    # Uses all 60K MNIST training points
    # CNN with KF training with rbf kernel
    X0 = tf.placeholder(tf.float32, [None, 28, 28])
    Y_ = tf.placeholder(tf.float32, [None, 10])
    step = tf.placeholder(tf.int32)
    pkeep = tf.placeholder(tf.float32)
    pkeep2 = tf.placeholder(tf.float32)
    tst = tf.placeholder(tf.bool)

    ba_size = tf.placeholder(tf.float32)

    X = tf.reshape(X0, [-1, 28, 28, 1])
    # Y_ = tf.one_hot(Y_id, 10)

    H = 150
    I = 150
    J = 150
    K = 300
    L = 300
    M = 300
    full_layer_size = 300
    N2 = 1200
    N = full_layer_size
    width = N
    in_fac = 1
    bias_init = 0.01

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.9,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    W1 = tf.Variable(tf.random.uniform([3, 3, 1, H],
                                       minval=-tf.sqrt(6 / (9 * (1 + H))),
                                       maxval=tf.sqrt(6 / (9 * (1 + H)))))
    B1 = tf.Variable(tf.constant(bias_init, tf.float32, [H]))
    W2 = tf.Variable(tf.random.uniform([3, 3, H, I],
                                       minval=-tf.sqrt(6 / (9 * (H + I))),
                                       maxval=tf.sqrt(6 / (9 * (H + I)))))
    B2 = tf.Variable(tf.constant(bias_init, tf.float32, [I]))
    W3 = tf.Variable(tf.random.uniform([5, 5, I, J],
                                       minval=-tf.sqrt(6 / (25 * (I + J))),
                                       maxval=tf.sqrt(6 / (25 * (I + J)))))
    B3 = tf.Variable(tf.constant(bias_init, tf.float32, [J]))
    W4 = tf.Variable(tf.random.uniform([3, 3, J, K],
                                       minval=-tf.sqrt(6 / (9 * (J + K))),
                                       maxval=tf.sqrt(6 / (9 * (J + K)))))
    B4 = tf.Variable(tf.constant(bias_init, tf.float32, [K]))
    W5 = tf.Variable(tf.random.uniform([3, 3, K, L],
                                       minval=-tf.sqrt(6 / (9 * (K + L))),
                                       maxval=tf.sqrt(6 / (9 * (K + L)))))
    B5 = tf.Variable(tf.constant(bias_init, tf.float32, [L]))
    W6 = tf.Variable(tf.random.uniform([5, 5, L, M],
                                       minval=-tf.sqrt(6 / (25 * (L + M))),
                                       maxval=tf.sqrt(6 / (25 * (L + M)))))
    B6 = tf.Variable(tf.constant(bias_init, tf.float32, [M]))
    # W15 = tf.Variable(tf.random.uniform([28 * 28 * J, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 28 * 28 * J)),
    #                                   maxval=tf.sqrt(0.01 / (N + 28 * 28 * J))))
    # W25 = tf.Variable(tf.random.uniform([14 * 14 * K, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * K)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * K))))
    # W35 = tf.Variable(tf.random.uniform([14 * 14 * L, N],
    #                                   minval=-tf.sqrt(0.01 / (N + 14 * 14 * L)),
    #                                   maxval=tf.sqrt(0.01 / (N + 14 * 14 * L))))
    W7 = tf.Variable(tf.random.uniform([16 * M, N],
                                       minval=-tf.sqrt(6 / (N + 16 * M)),
                                       maxval=tf.sqrt(6 / (N + 16 * M))))
    B7 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))

    W8 = tf.Variable(tf.random.uniform([N, N2], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B8 = tf.Variable(tf.constant(bias_init, tf.float32, [N2]))
    W9 = tf.Variable(tf.random.uniform([N2, N], minval=-tf.sqrt(6 / (N + N2)), maxval=tf.sqrt(6 / (N + N2))))
    B9 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    # W58 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(0.01 / (N + 10)), maxval=tf.sqrt(0.01 / (N + 10))))
    # W68 = tf.Variable(tf.random.uniform([N2, 10], minval=-tf.sqrt(0.01 / (N2 + 10)), maxval=tf.sqrt(0.01 / (N2 + 10))))

    W10 = tf.Variable(tf.random.uniform([N, 10], minval=-tf.sqrt(6 / (N + 10)), maxval=tf.sqrt(6 / (N + 10))))
    B10 = tf.Variable(tf.constant(bias_init, tf.float32, [10]))

    W37 = tf.Variable(tf.random.uniform([144 * J, N],
                                        minval=-tf.sqrt(6 / 144 * J + N),
                                        maxval=tf.sqrt(6 / 144 * J + N)))
    B37 = tf.Variable(tf.constant(bias_init, tf.float32, [N]))
    s37 = tf.Variable(tf.constant(1, tf.float32, [1, J]))



    s = tf.Variable(tf.constant([2, 2], tf.float32, [2]))

    # stride = 1  # output is 28x28
    # Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    # stride = 2  # output is 14x14
    # Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    # stride = 2  # output is 7x7
    # Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    stride = 1  # output is 28x28
    # Y0 = tf.nn.relu(tf.nn.conv2d(X, W0, strides=[1, stride, stride, 1], padding='SAME') + B0)

    Y1a = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='VALID') + B1
    Y1b, update_ema1 = batchnorm(Y1a, tst, step, B1, convolutional=True)
    Y1c = tf.nn.relu(Y1b)
    Y1 = tf.nn.dropout(Y1c, pkeep)

    Y2a = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='VALID') + B2
    Y2b, update_ema2 = batchnorm(Y2a, tst, step, B2, convolutional=True)
    Y2c = tf.nn.relu(Y2b)
    Y2 = tf.nn.dropout(Y2c, pkeep)

    Y3a = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3
    Y3b, update_ema3 = batchnorm(Y3a, tst, step, B3, convolutional=True)
    Y3c = tf.nn.relu(Y3b)
    Y3d = tf.nn.dropout(Y3c, pkeep)
    Y3 = tf.nn.max_pool2d(Y3d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    Y4a = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='VALID') + B4
    Y4b, update_ema4 = batchnorm(Y4a, tst, step, B4, convolutional=True)
    Y4c = tf.nn.relu(Y4b)
    Y4 = tf.nn.dropout(Y4c, pkeep)

    Y5a = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='VALID') + B5
    Y5b, update_ema5 = batchnorm(Y5a, tst, step, B5, convolutional=True)
    Y5c = tf.nn.relu(Y5b)
    Y5 = tf.nn.dropout(Y5c, pkeep)

    Y6a = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME') + B6
    Y6b, update_ema6 = batchnorm(Y6a, tst, step, B6, convolutional=True)
    Y6c = tf.nn.relu(Y6b)
    Y6d = tf.nn.dropout(Y6c, pkeep)
    Y6 = tf.nn.max_pool2d(Y6d, ksize=(2, 2), strides=(2, 2), padding='SAME')

    # reshape the output from the third convolution for the fully connected layer
    # YY1 = tf.reshape(Y1, shape=[-1, 24 * 24 * J])
    # YY2 = tf.reshape(Y2, shape=[-1, 12 * 12 * K])
    # YY3 = tf.reshape(Y3, shape=[-1, 12 * 12 * L])
    Y7 = tf.reduce_mean(tf.reshape(Y6, shape=[-1, 16, M]), axis=1)
    #Y7 = tf.reduce_mean(tf.reshape(Y6d, shape=[-1, 64, M]), axis=1)

    # Y5 = tf.matmul(YY1, W15) + tf.matmul(YY2, W25) + tf.matmul(YY3, W35) + tf.matmul(YY, W5) + B5
    #Y7 = tf.matmul(YY, W7) + B7
    Y37 = tf.reduce_mean(tf.reduce_mean(Y3, axis=1), axis=1)#tf.matmul(tf.reshape(Y3, shape=[-1, 144 * J]), W37) + B37



    Y8a = tf.matmul(Y7, W8) + B8
    Y8b, update_ema8 = batchnorm(Y8a, tst, step, B8, convolutional=False)
    Y8c = tf.nn.relu(Y8b)
    Y8 = tf.nn.dropout(Y8c, pkeep2)

    Y9a = tf.matmul(Y8, W9) + B9
    Y9b, update_ema9 = batchnorm(Y9a, tst, step, B9, convolutional=False)
    Y9c = tf.nn.relu(Y9b)
    Y9 = tf.nn.dropout(Y9c, pkeep2)

    Ylogits = tf.matmul(Y9, W10) + B10

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, update_ema6, update_ema8,
                          update_ema9)

    s = tf.maximum(s, 0.3)

    #kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    #weight = s[0] / tf.reduce_mean(kernel0)
    #kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sim_mat = tf.reduce_sum(tf.reshape(Y_ + 0.1, (-1, 1, 10)) * tf.reshape(Y_ + 0.1, (1, -1, 10)), axis=2)
    all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
    same_mat = sim_mat * all_mat
    diff_mat = all_mat * (1 - sim_mat)

    # kernel0 = tf.reduce_sum(tf.square(tf.reshape(Y3, (-1, 1, 100, 144)) - tf.reshape(Y3, (1, -1, 100, 144))), axis=3)

    # weight = s[0] / tf.reduce_mean(kernel0)
    # kernel0 = tf.reduce_mean(tf.exp(-weight * kernel0), axis=2)

    sq_YY0 = tf.reduce_sum(tf.square(Y37), axis=1)
    sq_YY0 = tf.reshape(sq_YY0, (-1, 1))

    pdist0 = -2 * tf.matmul(Y37, tf.transpose(Y37))
    pdist0 = pdist0 + sq_YY0
    pdist0 = pdist0 + tf.transpose(sq_YY0)
    kernel0 = tf.exp(-s[0] * pdist0 / tf.reduce_mean(pdist0))

    sq_YY = tf.reduce_sum(tf.square(Y7), axis=1)
    sq_YY = tf.reshape(sq_YY, (-1, 1))

    pdist = -2 * tf.matmul(Y7, tf.transpose(Y7))
    pdist = pdist + sq_YY
    pdist = pdist + tf.transpose(sq_YY)
    kernel = tf.exp(-s[1] * pdist / tf.reduce_mean(pdist))

    presig = tf.range(tf.shape(Y_)[0])
    # presig = tf.random_shuffle(presig)
    sig = presig[:tf.shape(Y_)[0] // 2]  # Gets the indicies of a random sample of half of the batch/data
    sig1 = presig[tf.shape(Y_)[0] // 2:]

    pi = tf.one_hot(sig, tf.shape(Y_)[0])
    pi1 = tf.one_hot(sig1, tf.shape(Y_)[0])

    smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    smallkernel1 = tf.matmul(pi1, tf.matmul(kernel, tf.linalg.matrix_transpose(pi1)))

    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff = tf.linalg.solve(smallkernel, Y_small)
    coeff1 = tf.linalg.solve(smallkernel1, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
    test_kernel = kernel[:, :tf.shape(Y_)[0] // 2]
    test_kernel1 = kernel[:, tf.shape(Y_)[0] // 2:]

    net_class = tf.matmul(test_kernel, coeff)
    net_class1 = tf.matmul(test_kernel1, coeff1)

    test_class = tf.matmul(pi_full, Y_)

    class_error = tf.reduce_sum(tf.square(net_class - test_class))
    class_error += tf.reduce_sum(tf.square(net_class1 - test_class))

    norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

    ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
    A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
    norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

    rho2 = 1 - norm_sig / norm_dagger

    smallkernel0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    smallkernel01 = tf.matmul(pi1, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi1)))
    Y_small = tf.matmul(pi, Y_)
    Y_small1 = tf.matmul(pi1, Y_)

    coeff0 = tf.linalg.solve(smallkernel0, Y_small)
    coeff01 = tf.linalg.solve(smallkernel01, Y_small1)

    pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

    kernel0 = tf.matmul(pi_full, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi_full)))
    test_kernel0 = kernel0[:, :tf.shape(Y_)[0] // 2]
    test_kernel01 = kernel0[:, tf.shape(Y_)[0] // 2:]

    net_class0 = tf.matmul(test_kernel0, coeff0)
    net_class01 = tf.matmul(test_kernel01, coeff01)
    test_class0 = tf.matmul(pi_full, Y_)

    class_error0 = tf.reduce_sum(tf.square(net_class0 - test_class0)) + tf.reduce_sum(
        tf.square(net_class01 - test_class0))

    norm_dagger0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel0, Y_)))

    ker_bar0 = tf.matmul(pi, tf.matmul(kernel0, tf.linalg.matrix_transpose(pi)))
    A_til0 = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar0, pi))
    norm_sig0 = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til0, Y_)))

    rho20 = 1 - norm_sig0 / norm_dagger0

    same_dist = tf.reduce_sum(pdist * same_mat) / tf.reduce_sum(same_mat)
    diff_dist = tf.reduce_sum(pdist * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist = tf.reduce_sum(pdist * all_mat) / tf.reduce_sum(all_mat)
    same_dist0 = tf.reduce_sum(pdist0 * same_mat) / tf.reduce_sum(same_mat)
    diff_dist0 = tf.reduce_sum(pdist0 * diff_mat) / tf.reduce_sum(diff_mat)
    all_dist0 = tf.reduce_sum(pdist0 * all_mat) / tf.reduce_sum(all_mat)

    dists = [same_dist, diff_dist, all_dist, same_dist0, diff_dist0, all_dist0]

    # mse = tf.reduce_mean(tf.square(Ylogits - Y_))
    mse = tf.reduce_mean(tf.square(Ylogits - (Y_ + 0.1)))

    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits + 0.1, labels=(Y_ + 0.1))
    cross_entropy = tf.reduce_mean(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    lr = 0.0000001 + tf.train.exponential_decay(0.001, step, 1000, 1 / math.e)  # cut lr in future
    train_step = tf.train.AdamOptimizer(lr).minimize(0.01 * class_error0 + 0.1 * class_error + cross_entropy)
    #train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
    #train_step = tf.train.MomentumOptimizer(10 * lr, 0.9).minimize(0.1 * class_error + cross_entropy)
    train_step1 = tf.train.AdamOptimizer(lr).minimize(class_error, var_list=[W1, B1, W2, B2, W3, B3, W4, B4])
    train_step2 = tf.train.AdamOptimizer(lr).minimize(mse, var_list=[W5, B5, W6, B6, W7, B7])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    acc_max = 0
    a = 0.3
    b = 0.3
    r = 0.1
    bs = 100
    err = np.zeros((121, 2, 2))
    rhoce = np.zeros((121, 3))
    distances = np.zeros((121, 6))
    for i in range(10001):
        # batch_X, batch_Y = get_batch(i)
        # if i < 5000: batch_X, batch_Y = get_rand_batch3(i, 600, tr_im, tr_ohid)
        # else: batch_X, batch_Y = get_rand_batch3(i, 100, tr_im, tr_ohid)
        # batch_size = int(150 + 2 * math.floor(375 * math.exp(-i / 1000.0)))
        # batch_size = 300

        # batch_X, batch_Y = get_def_rand_batch2(i, batch_size, tr_im, tr_ohid)
        batch_X, batch_Y = get_rand_batch40(bs)
        # for i in range(10):
        #    plt.imshow(batch_X[i].reshape((28, 28)))
        #    plt.show()
        # batch_X, batch_Y = get_rand_batch4(batch_size)

        sess.run(update_ema,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})
        sess.run(train_step,
                 {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0, pkeep2: 1.0, tst: False, ba_size: bs})

        if i % 100 == 0:
            sig, rho1, ce1, ce0, cr1, ds = sess.run([s, rho2, class_error, class_error0, cross_entropy, dists],
                                                {X0: batch_X, Y_: batch_Y, step: i, tst: False, pkeep: 1.0, pkeep2: 1.0,
                                                 ba_size: bs})
            distances[i // 100] = np.asarray(ds)
            rhoce[i // 100, 0] = rho1
            rhoce[i // 100, 1] = ce0
            rhoce[i // 100, 2] = ce1
            print(i, "rho: ", rho1, " class error: ", ce0, ce1, "cross ent:", cr1, "sig:", sig, "dists:", distances[i // 100], time.localtime())
        pkeep_i = 0.75  # + 0.25 * (math.exp(-(i - 0) / 1000))
        # if i < 5000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step2, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # if i < 3000: sess.run(train_step1, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})
        # else: sess.run(train_step, {X0: batch_X, Y_: batch_Y, step: i, pkeep: 1.0})

        if i % 1000 == 0 and i >= 0:
            acc = np.zeros((20))
            c = np.zeros((20))
            for j in range(20):
                acc[j], c[j] = sess.run([accuracy, cross_entropy],
                                        {X0: te_im[j * 500: (j + 1) * 500],
                                         Y_: te_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                         ba_size: bs})

            err[i // 100, 0, 0] = 1 - acc.mean()
            err[i // 100, 0, 1] = c.mean()
            # a, c = sess.run([accuracy, cross_entropy], {X0: te_im, Y_id: te_id, pkeep: 1.0})
            if acc.mean() > acc_max:
                acc_max = acc.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* test accuracy:" + str(
                acc.mean()) + " test loss: " + str(c.mean()) + " max accuracy:" + str(acc_max))

        if i == 12000:
            acct = np.zeros((120))
            ct = np.zeros((120))
            for j in range(120):
                acct[j], ct[j] = sess.run([accuracy, cross_entropy],
                                          {X0: tr_im[j * 500: (j + 1) * 500],
                                           Y_: tr_ohid[j * 500: (j + 1) * 500], pkeep: 1.0, pkeep2: 1.0, tst: True,
                                           ba_size: bs})
            err[i // 100, 1, 0] = 1 - acct.mean()
            err[i // 100, 1, 1] = ct.mean()
            print(str(i) + ": ********* epoch " + str(
                i * 100 // N_tr + 1) + " ********* train accuracy:" + str(
                acct.mean()) + " train loss: " + str(ct.mean()))

    return acc.mean(), acct.mean(), acc_max, err, rhoce, distances


