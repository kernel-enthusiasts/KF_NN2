import CNN_MNIST_Val
import numpy as np
n_tr = 5
acc_list = np.zeros((n_tr))
acct_list = np.zeros((n_tr))
p1 = 0.7
p2 = 0.7
for i in range(n_tr):
    acc_list[i], acct_list[i], *_ = CNN_MNIST_Val.CNN_KF18do_RBF_AUG(pk=p1, pk2=p2)
    print(acc_list)
    print(acct_list)

print(acc_list.mean(), acct_list.mean(), p1, p2)