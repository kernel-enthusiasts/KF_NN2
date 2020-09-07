import CNN_CIFAR10_Val

kf_fac = 3.0
gf = 1.5

CNN_CIFAR10_Val.WRN_KF(KFF=kf_fac, g_fac=gf)

print(kf_fac, gf)