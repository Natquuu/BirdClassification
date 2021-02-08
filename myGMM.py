import numpy as np


class GMM:
    """Gaussian Mixture. diag

        Representation of a Gaussian mixture model probability distribution.
        This class allows to estimate the parameters of a Gaussian mixture
        distribution.

        Parameters
    ----------
        n_components: number of mixture components, default=1

        tolerance : convergence threshold. EM iterations will stop when the
                    -loglikelihood is below this threshold, default=0.001

        max_iter : number of EM iterations, default=100

        Attributes
        ----------
        weights_ : weights of each mixture components.
                   Array-like of shape (n_components,)

        means_ : mean of each mixture component.
                 Array-like of shape (n_components, n_features)

        covariances_ : covariance of each mixture component.
                       Array-like of shape (n_components, n_features)

        precisions_ : precision matrices for each component
            in the mixture. A precision matrix is the inverse of
            a covariance matrix. A covariance matrix is symmetric positive
            definite so the mixture of Gaussian can be equivalently
            parameterized by the precision matrices. Storing the
            precision matrices instead of the covariance matrices makes
            it more efficient to compute the log-likelihood of new
            samples at test time. Array-like of shape
            (n_components, n_features)

        converged_ : when convergence was reached in
                     fit() is True, otherwise False.

        n_iter_ : number of step used by the
                  best fit of EM to reach the convergence.

        lower_bound_ : lower bound value on the log-likelihood
                       (of the training data with respect to the model)
                       of the best fit of EM.
        """

    def __init__(self, n_components=1, max_iter=100, tolerance=0.001):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tolerance = tolerance

    def fit(self, X):
        # # step 1 initialize
        # self._initialize_parameters(X)
        # max_lower_bound = -np.infty
        # n_samples, _ = X.shape
        # lower_bound = (-np.infty if do_init else self.lower_bound_)
        # for n_iter in range(1, self.max_iter + 1):
        # self.prev_lower_bound = lower_bound
        #     log_prob_norm, log_resp = self._e_step(X)
        #     self._m_step(X, log_resp)
        #     lower_bound = self._compute_lower_bound(
        #         log_resp, log_prob_norm)

        #     change = lower_bound - prev_lower_bound
        #     self._print_verbose_msg_iter_end(n_iter, change)

        #     if abs(change) < self.tol:
        #         self.converged_ = True
        #         break

        #     self._print_verbose_msg_init_end(lower_bound)

        #     if lower_bound > max_lower_bound:
        #         max_lower_bound = lower_bound
        #         best_params = self._get_parameters()
        #         best_n_iter = n_iter

        # if not self.converged_:
        #     pass
        # warnings.warn('Initialization %d did not converge. '
        #             'Try different init parameters, '
        #             'or increase max_iter, tol '
        #             'or check for degenerate data.'
        #             % (init + 1), ConvergenceWarning)

        # self._set_parameters(best_params)
        # self.n_iter_ = best_n_iter
        # self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        # _, log_resp = self._e_step(X)

        # return log_resp.argmax(axis=1)
        pass

    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        pass

    def _estimate_gaussian_parameters(self, X, resp):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
        avg_means2 = means ** 2
        avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
        covar = avg_X2 - 2 * avg_X_means + avg_means2

        return nk, means, covar

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self._estimate_gaussian_parameters(
            X, resp)
        weights /= n_samples

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances

    def _initialize_parameters(self, X):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        random_state = np.random.mtrand._rand
        n_samples, _ = X.shape
        resp = random_state.rand(n_samples, self.n_components)
        resp /= resp.sum(axis=1)[:, np.newaxis]
        self._initialize(X, resp)
