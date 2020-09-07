from __future__ import absolute_import, division, print_function, unicode_literals
# from import_mnist import imload
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import math
import time
import numpy.linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from sortedcontainers import SortedList

import resnet10021
import resnet10021_do

physical_devices = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

logical_devices = tf.config.experimental.list_logical_devices('GPU')

tr_num = 10011
get_tr_err = False

datagen = IDG(rotation_range=0,
            zoom_range = 0.0,
            width_shift_range=4,
            height_shift_range=4,
            horizontal_flip=True,
            vertical_flip=False,
            data_format="channels_last",
            zca_whitening=False,
            fill_mode='reflect')

(tr_im0, tr_id), (_, _) = tf.keras.datasets.cifar100.load_data()
tr_im0 = np.asarray(tr_im0, dtype=np.float32)
#te_im0 = np.asarray(te_im0, dtype=np.float32)
tr_id = np.asarray(tr_id, dtype=np.int32).reshape((-1))
#te_id = np.asarray(te_id, dtype=np.int32).reshape((-1))

te_im0 = tr_im0[40000:]
te_id = tr_id[40000:]
tr_im0 = tr_im0[:40000]
tr_id = tr_id[:40000]

cifar_mean = np.zeros((1, 1, 1, 3))
cifar_std = np.zeros((1, 1, 1, 3))
cifar_mean[0, 0, 0] = [tr_im0[:, :, :, 0].mean(), tr_im0[:, :, :, 1].mean(), tr_im0[:, :, :, 2].mean()]
cifar_std[0, 0, 0] = [tr_im0[:, :, :, 0].std(), tr_im0[:, :, :, 1].std(), tr_im0[:, :, :, 2].std()]

tr_im = (tr_im0 - cifar_mean) / cifar_std
te_im = (te_im0 - cifar_mean) / cifar_std


tr_N = 40000
num_cl = 100

tr_vec = tr_im.reshape((tr_N, 3072))
te_vec = te_im.reshape((10000, 3072))

tr_ohid = np.zeros((tr_N, 100))
te_ohid = np.zeros((10000, 100))

for i in range(tr_N):
    tr_ohid[i, int(tr_id[i])] = 1
    if i < 10000:
        te_ohid[i, int(te_id[i])] = 1

digit_dict = {}

for i in range(100):
    digit_dict[i] = SortedList()

for i in range(tr_N):
    digit_dict[int(tr_id[i])].add(i)

def get_def_rand_batch185(kfbs, dps, batch_size):
    batch_indx1 = np.random.choice(tr_N, batch_size, replace=False)

    test_indx = batch_indx1[:kfbs]

    #batch_X = tr_im[test_indx]
    batch_Y = tr_ohid[test_indx]

    kf_indx = np.zeros((0))
    cl_batch = np.sum(batch_Y, axis=0)
    tot_cl = np.count_nonzero(cl_batch)

    batch_indx1 = batch_indx1[:-(tot_cl * dps)]

    #batchkf1_X = tr_im[batch_indx1]
    batchkf1_Y = tr_ohid[batch_indx1]

    #print(tr_id[test_indx])
    #print(test_indx)

    for i in range(cl_batch.shape[0]):
        if cl_batch[i] > 0:
            digits_i = copy.deepcopy(digit_dict[i])
            for j in range(batch_indx1.shape[0]):
                if batchkf1_Y[j, i] > 0.5:
                    digits_i.remove(batch_indx1[j])
            kf_indx = np.concatenate([kf_indx, np.random.choice(digits_i, dps, replace=False)], axis=0)
            #print(i, digits_i)

    kf_indx = np.concatenate([batch_indx1, kf_indx], axis=0)
    kf_indx = np.asarray(kf_indx, dtype=np.int)

    #q = np.zeros(50000, dtype=int)
    #q[kf_indx] = 1

    #if q.sum() != batch_size:
    #    print(q.sum())
    #    print(kf_indx)

    kf_batch_X = tr_im0[kf_indx]
    kf_batch_Y = tr_ohid[kf_indx]

    for indx in range(batch_size):
        state = np.random.randint(5, size=1)[0]
        if 0 <= state <= 4:
            def_im = datagen.flow(kf_batch_X[indx].reshape((1, 32, 32, 3)), batch_size=1)
            kf_batch_X[indx] = def_im[0][0].reshape((32, 32, 3))
    kf_batch_X = (kf_batch_X - cifar_mean) / cifar_std

    return kf_batch_X, np.argmax(kf_batch_Y, axis=1).reshape((-1)), kf_indx

def get_def_rand_batch19(kfbs, dps, batch_size):
    batch_indx1 = np.random.choice(tr_N, batch_size, replace=False)

    test_indx = batch_indx1[:kfbs]

    #batch_X = tr_im[test_indx]
    batch_Y = tr_ohid[test_indx]

    kf_indx = np.zeros((0))
    cl_batch = np.sum(batch_Y, axis=0)
    tot_cl = np.count_nonzero(cl_batch)

    batch_indx1 = batch_indx1[:-(tot_cl * dps)]

    #batchkf1_X = tr_im[batch_indx1]
    batchkf1_Y = tr_ohid[batch_indx1]

    #print(tr_id[test_indx])
    #print(test_indx)

    for i in range(cl_batch.shape[0]):
        if cl_batch[i] > 0:
            digits_i = copy.deepcopy(digit_dict[i])
            for j in range(batch_indx1.shape[0]):
                if batchkf1_Y[j, i] > 0.5:
                    digits_i.remove(batch_indx1[j])
            kf_indx = np.concatenate([kf_indx, np.random.choice(digits_i, dps, replace=False)], axis=0)
            #print(i, digits_i)

    kf_indx = np.concatenate([batch_indx1, kf_indx], axis=0)
    kf_indx = np.asarray(kf_indx, dtype=np.int)

    #q = np.zeros(50000, dtype=int)
    #q[kf_indx] = 1

    #if q.sum() != batch_size:
    #    print(q.sum())
    #    print(kf_indx)

    kf_batch_X = tr_im0[kf_indx]
    kf_batch_Y = tr_ohid[kf_indx]

    for indx in range(batch_size):
        state = np.random.randint(5, size=1)[0]
        if 0 <= state <= 4:
            def_im = datagen.flow(kf_batch_X[indx].reshape((1, 32, 32, 3)), batch_size=1)
            kf_batch_X[indx] = def_im[0][0].reshape((32, 32, 3))

    kf_batch_X = (kf_batch_X - cifar_mean) / cifar_std
    return kf_batch_X, np.argmax(kf_batch_Y, axis=1).reshape((-1))


def get_rand_batch19(kfbs, dps, batch_size):
    batch_indx1 = np.random.choice(tr_N, batch_size, replace=False)

    test_indx = batch_indx1[:kfbs]

    #batch_X = tr_im[test_indx]
    batch_Y = tr_ohid[test_indx]

    kf_indx = np.zeros((0))
    cl_batch = np.sum(batch_Y, axis=0)
    tot_cl = np.count_nonzero(cl_batch)

    batch_indx1 = batch_indx1[:-(tot_cl * dps)]

    #batchkf1_X = tr_im[batch_indx1]
    batchkf1_Y = tr_ohid[batch_indx1]

    for i in range(cl_batch.shape[0]):
        if cl_batch[i] > 0:
            digits_i = copy.deepcopy(digit_dict[i])
            for j in range(batch_indx1.shape[0]):
                if batchkf1_Y[j, i] > 0.5:
                    digits_i.remove(batch_indx1[j])
            kf_indx = np.concatenate([kf_indx, np.random.choice(digits_i, dps, replace=False)], axis=0)

    kf_indx = np.concatenate([batch_indx1, kf_indx], axis=0)
    kf_indx = np.asarray(kf_indx, dtype=np.int)

    kf_batch_X = tr_im0[kf_indx]
    kf_batch_Y = tr_ohid[kf_indx]

    kf_batch_X = (kf_batch_X - cifar_mean) / cifar_std
    return kf_batch_X, np.argmax(kf_batch_Y, axis=1).reshape((-1))

def WRN_KF(KFF=0.0, g_fac=2.5):
    with tf.Graph().as_default():
        num_classes = 100
        num_train_instance = tr_N
        num_test_instance = 10000
        batch_size = 100
        bs = 3 * batch_size // 10
        num_residual_units = 16
        k = 8
        L2_weight = 0.0005
        momentum = 0.9
        initial_lr = 0.1
        lr_step_epoch = 60.0
        lr_decay = 0.2
        max_steps = int(200.0 * tr_N // batch_size) + 1
        display = 100
        test_interval = 10000
        test_iter = 100
        checkpoint_interval = 10000
        gpu_fraction = 0.95
        log_device_placement = False
        num_disp = (max_steps // display) + 1
        num_test = (max_steps // test_interval) + 1

        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels = tf.placeholder(tf.int32, [None])

        print("pre-build")
        # Build model
        decay_step = lr_step_epoch * num_train_instance / batch_size
        hp = resnet10021.HParams(batch_size=batch_size,
                                 bs = bs,
                             num_classes=num_classes,
                             num_residual_units=num_residual_units,
                             k=k,
                             initial_lr=initial_lr,
                             decay_step=decay_step,
                             lr_decay=lr_decay,
                             momentum=momentum)
        network = resnet10021.ResNet(hp, images, labels, global_step)
        network.build_model()
        network.build_train_op()

        # Summaries(training)
        train_summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        step_train = np.zeros((num_disp))
        step_test = np.zeros((num_test))
        te_loss_list = np.zeros((num_test))
        te_acc_list = np.zeros((num_test))
        train_acc_list = np.zeros((num_test))
        train_loss_list = np.zeros((num_test))

        gamma_list = np.zeros((num_disp, 4))
        rho_list = np.zeros((num_disp, 4))
        ce_list = np.zeros((num_disp, 4))
        tr_loss_list = np.zeros((num_disp))
        tr_acc_list = np.zeros((num_disp))
        lr_list = np.zeros((num_disp))

        # Training!
        switch = True
        test_best_acc = 0.0
        for step in range(init_step, max_steps):
            epoch = float(step) * float(batch_size) / float(num_train_instance)
            if epoch < 60.0:
                lrate = initial_lr
            elif epoch < 120.0:
                lrate = initial_lr * (lr_decay)
            elif epoch < 160.0:
                lrate = initial_lr * (lr_decay ** 2)
            else:
                lrate = initial_lr * (lr_decay ** 3)
            # Test
            if step < 500:
                kff = 0
                l2_weight = L2_weight / 1.0
            else:
                kff = KFF / 1.0
                l2_weight = L2_weight / 1.0
            if switch:
                try:
                    if step % test_interval == 0:
                        test_loss, test_acc = 0.0, 0.0
                        test_iter = num_test_instance // batch_size
                        for i in range(test_iter):
                            loss_value, acc_value = sess.run([network.loss, network.acc],
                                                             feed_dict={network.is_train: False, network.KF_factor: kff, network.weight_decay:l2_weight, network.lr: lrate,
                                                                        network.gam_factor: g_fac,
                                                                        images: te_im[i * batch_size:(i + 1) * batch_size].reshape(
                                                                            (-1, 32, 32, 3)),
                                                                        labels: te_id[i * batch_size:(i + 1) * batch_size]})
                            test_loss += loss_value
                            test_acc += acc_value
                        test_loss /= test_iter
                        test_acc /= test_iter
                        test_best_acc = max(test_best_acc, test_acc)
                        format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f, top acc=%.4f')
                        print(format_str % (time.localtime(), step, test_loss, test_acc, test_best_acc))
                        if step > 5000 and test_acc < 0.1:
                            switch = False
                        step_test[step // test_interval] = step
                        te_acc_list[step // test_interval] = test_acc
                        te_loss_list[step // test_interval] = test_loss
                        np.save("Ctest_stats_KF" + str(tr_num) + ".npy", [step_test, te_acc_list, te_loss_list])
                        if get_tr_err:
                            tr_iter = num_train_instance // batch_size
                            tr_loss, tr_acc = 0.0, 0.0
                            for i in range(tr_iter):
                                loss_value, acc_value = sess.run([network.loss, network.acc],
                                                                 feed_dict={network.is_train: False, network.KF_factor: kff, network.weight_decay:l2_weight, network.lr: lrate,
                                                                            network.gam_factor: g_fac,
                                                                            images: tr_im[i * batch_size:(i + 1) * batch_size].reshape(
                                                                                (-1, 32, 32, 3)),
                                                                            labels: tr_id[i * batch_size:(i + 1) * batch_size]})
                                tr_loss += loss_value
                                tr_acc += acc_value
                            tr_loss /= tr_iter
                            tr_acc /= tr_iter
                            format_str = ('%s: (Train)     step %d, loss=%.4f, acc=%.4f')
                            print(format_str % (time.localtime(), step, tr_loss, tr_acc))
                            step_test[step // test_interval] = step
                            train_acc_list[step // test_interval] = tr_acc
                            train_loss_list[step // test_interval] = tr_loss
                            np.save("Ctrain_acc_KF" + str(tr_num) + ".npy", [step_test, train_acc_list, train_loss_list])


                    train_images_val, train_labels_val, kfb = get_def_rand_batch185(bs, 2, batch_size)
                    _, lr_value, loss_value, acc_value, ce, rho, train_summary_str, gamma = \
                        sess.run([network.train_op, network.lr, network.loss, network.acc, network.class_error, network.rho2,
                                  train_summary_op, network.gam],
                                 feed_dict={network.is_train: True, images: train_images_val.reshape((-1, 32, 32, 3)), network.weight_decay:l2_weight, network.lr: lrate,
                                            labels: train_labels_val, network.gam_factor: g_fac, network.KF_factor: kff})
                    if step > 5000 and not 0 <= loss_value <= 3:
                        print(kfb)
                        print(train_labels_val)
                        print(lr_value, loss_value, acc_value, ce, rho, train_summary_str, gamma)

                    assert not np.isnan(loss_value)

                    # Display & Summary(training)
                    if step % display == 0:
                        step_train[step // display] = step
                        gamma_list[step // display] = np.asarray(gamma).reshape((4))
                        rho_list[step // display] = np.asarray(rho).reshape((4))
                        ce_list[step // display] = np.asarray(ce).reshape((4))
                        tr_loss_list[step // display] = loss_value
                        tr_acc_list[step // display] = acc_value
                        lr_list[step // display] = lr_value

                        np.save("Ctrain_stats_KF" + str(tr_num) + ".npy", [step_train, tr_loss_list, tr_acc_list, lr_list])
                        np.save("Ctrain_kf_stats_KF" + str(tr_num) + ".npy", [gamma_list, rho_list, ce_list])
                        print(time.localtime(), step, gamma, rho, ce, loss_value, acc_value, lr_value)
                except:
                    pass

def WRN_DO(KFF=0.0, g_fac=2.5):
    with tf.Graph().as_default():
        num_classes = 100
        num_train_instance = tr_N
        num_test_instance = 10000
        batch_size = 100
        bs = 3 * batch_size // 10
        num_residual_units = 16
        k = 8
        L2_weight = 0.0005
        momentum = 0.9
        initial_lr = 0.1
        lr_step_epoch = 60.0
        lr_decay = 0.2
        max_steps = int(200.0 * tr_N // batch_size) + 1
        display = 100
        test_interval = 10000
        test_iter = 100
        checkpoint_interval = 10000
        gpu_fraction = 0.95
        log_device_placement = False
        num_disp = (max_steps // display) + 1
        num_test = (max_steps // test_interval) + 1

        init_step = 0
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # Build a Graph that computes the predictions from the inference model.
        images = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels = tf.placeholder(tf.int32, [None])

        print("pre-build")
        # Build model
        decay_step = lr_step_epoch * num_train_instance / batch_size
        hp = resnet10021_do.HParams(batch_size=batch_size,
                                 bs = bs,
                             num_classes=num_classes,
                             num_residual_units=num_residual_units,
                             k=k,
                             initial_lr=initial_lr,
                             decay_step=decay_step,
                             lr_decay=lr_decay,
                             momentum=momentum)
        network = resnet10021_do.ResNet(hp, images, labels, global_step)
        network.build_model()
        network.build_train_op()

        # Summaries(training)
        train_summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        step_train = np.zeros((num_disp))
        step_test = np.zeros((num_test))
        te_loss_list = np.zeros((num_test))
        te_acc_list = np.zeros((num_test))
        train_acc_list = np.zeros((num_test))
        train_loss_list = np.zeros((num_test))

        gamma_list = np.zeros((num_disp, 4))
        rho_list = np.zeros((num_disp, 4))
        ce_list = np.zeros((num_disp, 4))
        tr_loss_list = np.zeros((num_disp))
        tr_acc_list = np.zeros((num_disp))
        lr_list = np.zeros((num_disp))

        # Training!
        switch = True
        test_best_acc = 0.0
        for step in range(init_step, max_steps):
            epoch = float(step) * float(batch_size) / float(num_train_instance)
            if epoch < 60.0:
                lrate = initial_lr
            elif epoch < 120.0:
                lrate = initial_lr * (lr_decay)
            elif epoch < 160.0:
                lrate = initial_lr * (lr_decay ** 2)
            else:
                lrate = initial_lr * (lr_decay ** 3)
            # Test
            if step < 500:
                kff = 0
                l2_weight = L2_weight / 1.0
            else:
                kff = KFF / 1.0
                l2_weight = L2_weight / 1.0
            if switch:
                try:
                    if step % test_interval == 0:
                        test_loss, test_acc = 0.0, 0.0
                        test_iter = num_test_instance // batch_size
                        for i in range(test_iter):
                            loss_value, acc_value = sess.run([network.loss, network.acc],
                                                             feed_dict={network.is_train: False, network.KF_factor: kff, network.weight_decay:l2_weight, network.lr: lrate,
                                                                        network.gam_factor: g_fac,
                                                                        images: te_im[i * batch_size:(i + 1) * batch_size].reshape(
                                                                            (-1, 32, 32, 3)),
                                                                        labels: te_id[i * batch_size:(i + 1) * batch_size]})
                            test_loss += loss_value
                            test_acc += acc_value
                        test_loss /= test_iter
                        test_acc /= test_iter
                        test_best_acc = max(test_best_acc, test_acc)
                        format_str = ('%s: (Test)     step %d, loss=%.4f, acc=%.4f, top acc=%.4f')
                        print(format_str % (time.localtime(), step, test_loss, test_acc, test_best_acc))
                        if step > 5000 and test_acc < 0.1:
                            switch = False
                        step_test[step // test_interval] = step
                        te_acc_list[step // test_interval] = test_acc
                        te_loss_list[step // test_interval] = test_loss
                        np.save("Ctest_stats_KF" + str(tr_num) + ".npy", [step_test, te_acc_list, te_loss_list])
                        if get_tr_err:
                            tr_iter = num_train_instance // batch_size
                            tr_loss, tr_acc = 0.0, 0.0
                            for i in range(tr_iter):
                                loss_value, acc_value = sess.run([network.loss, network.acc],
                                                                 feed_dict={network.is_train: False, network.KF_factor: kff, network.weight_decay:l2_weight, network.lr: lrate,
                                                                            images: tr_im[i * batch_size:(i + 1) * batch_size].reshape(
                                                                                (-1, 32, 32, 3)),
                                                                            labels: tr_id[i * batch_size:(i + 1) * batch_size]})
                                tr_loss += loss_value
                                tr_acc += acc_value
                            tr_loss /= tr_iter
                            tr_acc /= tr_iter
                            format_str = ('%s: (Train)     step %d, loss=%.4f, acc=%.4f')
                            print(format_str % (time.localtime(), step, tr_loss, tr_acc))
                            step_test[step // test_interval] = step
                            train_acc_list[step // test_interval] = tr_acc
                            train_loss_list[step // test_interval] = tr_loss
                            np.save("Ctrain_acc_KF" + str(tr_num) + ".npy", [step_test, train_acc_list, train_loss_list])


                    train_images_val, train_labels_val, kfb = get_def_rand_batch185(bs, 2, batch_size)
                    _, lr_value, loss_value, acc_value, ce, rho, train_summary_str, gamma = \
                        sess.run([network.train_op, network.lr, network.loss, network.acc, network.class_error, network.rho2,
                                  train_summary_op, network.gam],
                                 feed_dict={network.is_train: True, images: train_images_val.reshape((-1, 32, 32, 3)), network.weight_decay:l2_weight, network.lr: lrate,
                                            labels: train_labels_val, network.gam_factor: g_fac, network.KF_factor: kff})
                    if step > 5000 and not 0 <= loss_value <= 3:
                        print(kfb)
                        print(train_labels_val)
                        print(lr_value, loss_value, acc_value, ce, rho, train_summary_str, gamma)

                    assert not np.isnan(loss_value)

                    # Display & Summary(training)
                    if step % display == 0:
                        step_train[step // display] = step
                        gamma_list[step // display] = np.asarray(gamma).reshape((4))
                        rho_list[step // display] = np.asarray(rho).reshape((4))
                        ce_list[step // display] = np.asarray(ce).reshape((4))
                        tr_loss_list[step // display] = loss_value
                        tr_acc_list[step // display] = acc_value
                        lr_list[step // display] = lr_value

                        np.save("Ctrain_stats_KF" + str(tr_num) + ".npy", [step_train, tr_loss_list, tr_acc_list, lr_list])
                        np.save("Ctrain_kf_stats_KF" + str(tr_num) + ".npy", [gamma_list, rho_list, ce_list])
                        print(time.localtime(), step, gamma, rho, ce, loss_value, acc_value, lr_value)
                except:
                    pass
