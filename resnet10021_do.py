from collections import namedtuple

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np

import utils

HParams = namedtuple('HParams',
                     'batch_size, bs, num_classes, num_residual_units, k, '
                     'initial_lr, decay_step, lr_decay, '
                     'momentum')


class ResNet(object):
    def __init__(self, hp, images, labels, global_step):
        self._hp = hp  # Hyperparameters
        self._images = images  # Input image
        self._labels = labels
        self._global_step = global_step
        self.is_train = tf.placeholder(tf.bool)
        self.KF_factor = tf.placeholder(tf.float32)
        self.weight_decay = tf.placeholder(tf.float32)
        self.gam_factor = tf.placeholder(tf.float32)

    def build_model(self):
        pkeep_c = 0.3 * tf.cast(self.is_train, dtype=tf.float32)
        pkeep_fc = 0.0
        print('Building model')
        # Init. conv.
        print('\tBuilding unit: init_conv')
        x = utils._conv(self._images, 3, 16, 1, name='init_conv')

        # Residual Blocks
        filters = [16, 16 * self._hp.k, 32 * self._hp.k, 64 * self._hp.k]
        strides = [1, 2, 2]

        KF_list = []
        num_ker = 4


        for i in range(1, 4):
            # First residual unit
            with tf.variable_scope('unit_%d_0' % i) as scope:
                print('\tBuilding residual unit: %s' % scope.name)
                x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                x = utils._relu(x, name='relu_1')

                # Shortcut
                if filters[i - 1] == filters[i]:
                    if strides[i - 1] == 1:
                        shortcut = tf.identity(x)
                    else:
                        shortcut = tf.nn.max_pool(x, [1, strides[i - 1], strides[i - 1], 1],
                                                  [1, strides[i - 1], strides[i - 1], 1], 'VALID')
                else:
                    shortcut = utils._conv(x, 1, filters[i], strides[i - 1], name='shortcut')

                # Residual
                x = utils._conv(x, 3, filters[i], strides[i - 1], name='conv_1')
                x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
                x = utils._relu(x, name='relu_2')
                x = tf.nn.dropout(x, rate=pkeep_c)
                x = utils._conv(x, 3, filters[i], 1, name='conv_2')

                # Merge
                x = x + shortcut
                KF_list.append(tf.reduce_mean(x, [1, 2]))
            # Other residual units
            for j in range(1, self._hp.num_residual_units):
                with tf.variable_scope('unit_%d_%d' % (i, j)) as scope:
                    print('\tBuilding residual unit: %s' % scope.name)
                    # Shortcut
                    shortcut = x

                    # Residual
                    x = utils._bn(x, self.is_train, self._global_step, name='bn_1')
                    x = utils._relu(x, name='relu_1')
                    x = tf.nn.dropout(x, rate=pkeep_c)

                    x = utils._conv(x, 3, filters[i], 1, name='conv_1')
                    x = utils._bn(x, self.is_train, self._global_step, name='bn_2')
                    x = utils._relu(x, name='relu_2')
                    x = utils._conv(x, 3, filters[i], 1, name='conv_2')

                    # Merge
                    x = x + shortcut

        # Last unit
        with tf.variable_scope('unit_last') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = utils._bn(x, self.is_train, self._global_step)
            x = utils._relu(x)
            x = tf.reduce_mean(x, [1, 2])

        KF_list.append(x)

        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x_shape = x.get_shape().as_list()
            x = tf.reshape(x, [-1, x_shape[1]])
            x = utils._fc(x, self._hp.num_classes)

        self._logits = x

        Y_ = tf.one_hot(self._labels, self._hp.num_classes, dtype=tf.float32) - (1.0 / self._hp.num_classes)

        ce_list = []
        rho_list = []

        bs = self._hp.bs
        s = tf.Variable(tf.constant(1.0, tf.float32, [4]), trainable=False) * self.gam_factor


        for i in range(num_ker):
            Y_KF = KF_list[i]

            sq_YY = tf.reduce_sum(tf.square(Y_KF), axis=1)
            sq_YY = tf.reshape(sq_YY, (-1, 1))

            # sq_X = tf.reduce_sum(tf.square(tf.reshape(X, [-1, 784])), axis=1)
            # sq_X = tf.reshape(sq_X, (-1, 1))

            kernel = -2 * tf.matmul(Y_KF, tf.transpose(Y_KF))
            kernel = kernel + sq_YY
            kernel = kernel + tf.transpose(sq_YY)

            weight = s[i] / tf.reduce_mean(kernel)
            kernel = tf.exp(-weight * kernel) + 0.0001 * tf.eye(tf.shape(Y_)[0])

            # kernel = tf.square(tf.reshape(Y_KF, (-1, 1, 300)) - tf.reshape(Y_KF, (1, -1, 300)))

            # weight = 2.0 / tf.reduce_mean(kernel)
            # kernel = tf.reduce_sum(tf.exp(-weight * kernel), axis=2)

            presig = tf.range(tf.shape(Y_)[0])
            # presig = tf.random_shuffle(presig)
            sig = presig[bs:]  # Gets the indicies of a random sample of half of the batch/data

            pi = tf.one_hot(sig, tf.shape(Y_)[0])

            smallkernel = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
            Y_small = tf.matmul(pi, Y_)

            coeff = tf.linalg.solve(smallkernel, Y_small)

            pi_full = tf.one_hot(presig, tf.shape(Y_)[0])

            kernel = tf.matmul(pi_full, tf.matmul(kernel, tf.linalg.matrix_transpose(pi_full)))
            test_kernel = kernel[:, bs:]

            net_class = tf.matmul(test_kernel, coeff)

            test_class = tf.matmul(pi_full, Y_)

            ce_list.append(tf.reduce_mean(tf.square(net_class - test_class)))

            norm_dagger = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.linalg.solve(kernel, Y_)))

            ker_bar = tf.matmul(pi, tf.matmul(kernel, tf.linalg.matrix_transpose(pi)))
            A_til = tf.matmul(tf.linalg.matrix_transpose(pi), tf.linalg.solve(ker_bar, pi))
            norm_sig = tf.linalg.trace(tf.matmul(tf.linalg.matrix_transpose(Y_), tf.matmul(A_til, Y_)))

            rho_list.append(1 - norm_sig / norm_dagger)

        Y_KFf = KF_list[num_ker - 1]

        sq_YYf = tf.reduce_sum(tf.square(Y_KFf), axis=1)
        sq_YYf = tf.reshape(sq_YYf, (-1, 1))

        # sq_X = tf.reduce_sum(tf.square(tf.reshape(X, [-1, 784])), axis=1)
        # sq_X = tf.reshape(sq_X, (-1, 1))

        pdistf = -2 * tf.matmul(Y_KFf, tf.transpose(Y_KFf))
        pdistf = pdistf + sq_YYf
        pdistf = pdistf + tf.transpose(sq_YYf)

        sim_mat = tf.reduce_sum(tf.reshape(Y_ + 1.0 / self._hp.num_classes, (-1, 1, 10)) * tf.reshape(Y_ + 1.0 / self._hp.num_classes, (1, -1, 10)), axis=2)
        all_mat = tf.ones_like(sim_mat) - tf.eye(tf.shape(Y_)[0])
        same_mat = sim_mat * all_mat
        diff_mat = all_mat * (1 - sim_mat)

        self.same_dist = tf.reduce_sum(pdistf * same_mat) / tf.reduce_sum(same_mat)
        self.diff_dist = tf.reduce_sum(pdistf * diff_mat) / tf.reduce_sum(diff_mat)
        self.all_dist = tf.reduce_sum(pdistf * all_mat) / tf.reduce_sum(all_mat)


        self.class_error = ce_list
        self.rho2 = rho_list
        self.gam = s

        self.KF_loss = ce_list[0] * 0.1 + ce_list[1] * 0.3 + ce_list[2] * 1.0 + ce_list[3] * 3.0

        self._logits = x

        # Probs & preds & acc
        self.probs = tf.nn.softmax(x, name='probs')
        self.preds = tf.to_int32(tf.argmax(self._logits, 1, name='preds'))
        ones = tf.constant(np.ones([self._hp.batch_size]), dtype=tf.float32)
        zeros = tf.constant(np.zeros([self._hp.batch_size]), dtype=tf.float32)
        correct = tf.where(tf.equal(self.preds, self._labels), ones, zeros)
        self.acc = tf.reduce_mean(correct, name='acc')
        tf.summary.scalar('accuracy', self.acc)

        # Loss & acc
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self._labels)
        self.loss = tf.reduce_mean(loss, name='cross_entropy')
        tf.summary.scalar('cross_entropy', self.loss)

    def build_train_op(self):
        # Add l2 loss
        with tf.variable_scope('l2_loss'):
            costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
            # for var in tf.get_collection(utils.WEIGHT_DECAY_KEY):
            # tf.summary.histogram(var.op.name, var)
            l2_loss = tf.multiply(self.weight_decay, tf.add_n(costs))

        def f1():
            return tf.constant(1, dtype=tf.float32)

        def f2():
            return tf.constant(0, dtype=tf.float32)
        self.KF_gate = tf.cond(tf.less(self.KF_loss, 0.1), f1, f2) * tf.cond(tf.greater(self.KF_loss, 0.0), f1, f2)

        def kfl():
            return self.loss + l2_loss + self.KF_factor * self.KF_loss
        def kfl0():
            return self.loss + l2_loss

        self._total_loss = tf.cond(tf.greater(self.KF_gate, 0.5), kfl, kfl0)

        # Learning rate
        self.lr = tf.placeholder(tf.float32)

        tf.summary.scalar('learing_rate', self.lr)

        # Gradient descent step
        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        grads_and_vars = opt.compute_gradients(self._total_loss, tf.trainable_variables())
        # print('\n'.join([t.name for t in tf.trainable_variables()]))
        apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Batch normalization moving average update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            with tf.control_dependencies(update_ops + [apply_grad_op]):
                self.train_op = tf.no_op()
        else:
            self.train_op = apply_grad_op
