from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import roc_auc_score
from ManifoldParzenWindows import MPW
import numpy as np

NUM_TRAINING_PER_CLASS = 100
TRAINING_DIGITS = range(8)
TEST_DIGITS = [8, 9]
NUM_VALIDATION = 1000
NUM_TEST_DIGITS = 1000

mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

images, labels = mnist.train.images, mnist.train.labels

tr_images = np.concatenate([images[labels == digit][:NUM_TRAINING_PER_CLASS] for digit in TRAINING_DIGITS])
tr_labels = np.concatenate([labels[labels == digit][:NUM_TRAINING_PER_CLASS] for digit in TRAINING_DIGITS])

te_images, te_labels = mnist.test.images[:NUM_TEST_DIGITS], mnist.test.labels[:NUM_TEST_DIGITS]
va_images = mnist.test.images[NUM_TEST_DIGITS:(NUM_TEST_DIGITS + NUM_VALIDATION)]
va_labels = mnist.test.labels[NUM_TEST_DIGITS:(NUM_TEST_DIGITS + NUM_VALIDATION)]

true_va_cls = np.isin(va_labels, TEST_DIGITS)

# Optimize over d, sig2, and k
DATA_DIM = 784
MIN_D, MAX_D = 5, 8
SEARCH_SIZE = 2
MIN_SIG2 = 0.4
MIN_K, MAX_K = 45, 50

best_models = [None] * (MAX_D - MIN_D)
best_anlls = np.zeros(MAX_D - MIN_D)
best_overall_densities = np.zeros((NUM_VALIDATION, MAX_D - MIN_D))
best_aucs = [None] * (MAX_D - MIN_D)

# Optimize d using AUC
for d in range(MIN_D, MAX_D):
    print("d:", d)
    # Optimize sig2 and k using ANLL
    sig2 = np.random.rand(SEARCH_SIZE) * 3 + MIN_SIG2
    k = np.random.randint(MIN_K, MAX_K, size=SEARCH_SIZE)
    best_anll = float("inf")
    best_index = -1
    best_model = None
    best_densities = None
    for i in range(SEARCH_SIZE):
        print("k and sig2:", k[i], sig2[i])
        m = MPW()
        m.build_model(tr_images, d, k[i], sig2[i])
        current_densities = m.get_likelihoods(va_images)
        print("max density:", current_densities.max())
        current_anll = m.anll(current_densities)
        print("current anll:", current_anll)
        if current_anll < best_anll:
            best_anll = current_anll
            best_index = i
            best_model = m.get_model()
            best_densities = current_densities

    idx = d - MIN_D
    best_models[idx] = best_model
    best_anlls[idx] = best_anll
    best_overall_densities[:, idx] = best_densities

    # Get AUC for each d
    best_aucs[idx] = roc_auc_score(true_va_cls, np.max(best_densities) - best_densities)
    print("best auc for this d: ", best_aucs[idx])
