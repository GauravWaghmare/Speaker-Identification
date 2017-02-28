import numpy as np
from sklearn.lda import LDA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
# import feature_loader
import math
from sklearn.preprocessing import OneHotEncoder
import VQ
import applyLDA as al
import mlpy
from sklearn.lda import LDA
from sklearn import preprocessing
from sklearn.decomposition import PCA, FactorAnalysis
from keras.regularizers import l2, activity_l2
from scipy import linalg
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV



def load_data(direc, num_speakers):
    X, Y = VQ.getFeatureMatrix(direc, num_speakers)
    indices  = np.random.permutation(Y.shape[0])

    X = X[indices, :]
    Y = Y[indices]

    total_rows = Y.shape[0]
    train_data_rows = int(total_rows)

    train_x = X[0:train_data_rows+1, :]
    train_y = Y[0:train_data_rows+1]
    train_data = (train_x, train_y)

    return train_data


direc = "/home/manvi/Desktop/voicebiometric/Phoneme/trainset/"

num_speakers = 15
train_data = load_data(direc, num_speakers)

X_train = train_data[0]
y_train = train_data[1]


print X_train.shape

X = X_train

# Fit the models
n_features = 49
n_components = np.arange(0, n_features, 5)  # options for n_components


def compute_scores(X):
    pca = PCA()
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        fa_scores.append(np.mean(cross_val_score(fa, X)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X))


pca_scores, fa_scores = compute_scores(X)
n_components_pca = n_components[np.argmax(pca_scores)]
n_components_fa = n_components[np.argmax(fa_scores)]

pca = PCA(n_components='mle')
pca.fit(X)
n_components_pca_mle = pca.n_components_

print "best n_components by PCA CV = %d" % n_components_pca
print "best n_components by FactorAnalysis CV = %d" % n_components_fa
print "best n_components by PCA MLE = %d" % n_components_pca_mle