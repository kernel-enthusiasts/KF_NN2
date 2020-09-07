import CNN_CIFAR100_Val

kf_fac = 1.0
gf = 3.5

CNN_CIFAR100_Val.WRN_KF(KFF=kf_fac, g_fac=gf)

print(kf_fac, gf)