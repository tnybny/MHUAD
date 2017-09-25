from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np


class MPW(object):
    """
    Implementation Manifold Parzen Windows from [1].
    We use the same dataset as that in the paper for training and evaulation.
    [1] Vincent, Pascal, and Yoshua Bengio. "Manifold parzen windows."
    Advances in neural information processing systems. 2003.
    """
    def __init__(self):
        self.model = None

    @staticmethod
    def anll(densities):
        # helper function to compute average negative log likelihood
        return -np.mean(np.log(densities[densities != 0]))

    @staticmethod
    def plot_2d_density(X, densities, extent, title=""):
        """
        :param X: training points
        :param densities: Densities over the grid
        :param extent: grid extents
        :param title: title for the plot
        """
        plt.figure(figsize=(10, 8))
        point_size = .5
        full_title = "Manifold Density Estimation: " + title
        plt.title(full_title)

        plt.imshow(densities, extent=extent, origin='lower')
        plt.colorbar()

        plt.scatter(X[:, 0], X[:, 1], cmap=plt.cm.viridis, s=point_size)
        plt.axes().set_aspect('equal', 'datalim')

        plt.show()

    @staticmethod
    def local_gaussian(x, x_i, V_i, lambda_i, d, sig2):
        """
        :param x: numpy 1-d array, test vector of shape (n, )
        :param x_i: numpy 1-d array, training vector of shape (n, )
        :param V_i: numpy ndarray, d eigenvectors corresponding to ith training point of shape (n, d)
        :param lambda_i: numpy 1-d array, d eigenvalues corresponding to the ith training point of shape (d, )
        :param d: int, number of principal directions
        :param sig2: float, regularization hyperparameter
        :returns: Gaussian density
        """
        n = x.shape[0]
        r = d * np.log(2 * np.pi) + np.sum(np.log(lambda_i)) + (n - d) * np.log(sig2)
        # above line has fixed a suspected typo in original paper which added \sigma^2 to lambda_i again here
        q = (1.0 / sig2) * np.sum(np.square(x - x_i))
        for j in range(d):
            temp = ((1.0 / lambda_i[j]) - (1.0 / sig2)) * np.square(np.dot(V_i[:, j], (x - x_i)))
            q = q + temp
        return np.exp(-0.5 * (r + q))

    def build_model(self, X, d, k, sig2):
        """
        :param X: numpy ndarray, training set of shape (l, n)
        :param d: int, number of principal directions
        :param k: int, chosen number of neighbors
        :param sig2: float, regularization hyperparameter
        :returns model: dict, contains (X, V, lambdaVec, k, d, sig2) where V is an (l, n, d) tensor that
        collects all the eigenvectors and lambdaVec is a (l, d) matrix with all the eigenvalues
        """
        l, n = X.shape
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', leaf_size=30).fit(X)
        V = np.zeros((l, n, d))
        lambdaVec = np.zeros((l, d))

        [_, idx] = nbrs.kneighbors(X)

        for i in xrange(l):
            ids = idx[i]
            M = np.zeros((k, n))
            for j in range(ids.shape[0]):
                M[j, :] = X[ids[j]] - X[i]

            Ui, s, Vi = np.linalg.svd(M, full_matrices=False)

            s_d = s[0:d]
            V_d = Vi[0:d, :].T

            V[i, :, :] = V_d
            lambdaVec[i, :] = (np.square(s_d) / l) + sig2

        model = {"tr_set": X, "eigenVec": V, "lambda": lambdaVec, "nbrs": k, "dim": d, "var": sig2}
        self.model = model
        return

    def get_likelihood(self, x):
        """
        :param x: numpy 1-d array, test vector of shape (n, )
        :returns phat: estimated density function evaluated at test point x
        """
        if self.model is None:
            return
        phat = 0
        X, V, lambdaVec, k, d, sig2 = self.model["tr_set"], self.model["eigenVec"], self.model["lambda"],\
            self.model["nbrs"], self.model["dim"], self.model["var"]
        l, n = X.shape
        for i in xrange(l):
            phat = phat + self.local_gaussian(x, X[i], V[i], lambdaVec[i], d, sig2)
        return phat / l

    def get_likelihoods(self, X_te):
        """
        :param X_te: numpy n-d array, test matrix of shape (num_test, n)
        :returns likelihoods: estimated density function evaluated at X_te
        """
        num_test, _ = X_te.shape
        likelihoods = np.array([self.get_likelihood(X_te[i]) for i in range(num_test)])
        return likelihoods

    def get_model(self):
        return self.model
