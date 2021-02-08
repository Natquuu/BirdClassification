import numpy as np
from scipy.stats import norm
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib


def plot_distributions(data, data_sampled, mu, sigma, K, color="green", color_sampled="red", name='plot.png'):
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 16})
    data_sampled = np.clip(data_sampled, np.min(data), np.max(data))
    plt.hist(data, bins=15, color=color, alpha=0.45, density=True)
    plt.hist(data_sampled, bins=15, range=(np.min(data), np.max(data)), color=color_sampled, alpha=0.45, density=True)
    for k in range(K):
        curve = np.linspace(mu[k] - 10 * sigma[k], mu[k] + 10 * sigma[k], 100)
        color = np.random.rand(3)
        plt.plot(curve, stats.norm.pdf(curve, mu[k], sigma[k]), color=color, linestyle="--", linewidth=3)
    plt.ylabel(r"$p(x)$")
    plt.xlabel(r"$x$")
    plt.tight_layout()
    plt.xlim(20, 120)
    plt.savefig(name, dpi=200)
    plt.show()


def plot_likelihood(nll_list):
    matplotlib.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 16})
    plt.plot(np.arange(len(nll_list)), nll_list, color="black", linestyle="--", linewidth=3)
    plt.ylabel(r"(negative) log-likelihood")
    plt.xlabel(r"iteration")
    plt.tight_layout()
    plt.xlim(0, len(nll_list))
    plt.savefig('nll.png', dpi=200)
    plt.show()


def sampler(pi, mu, sigma, N):
    data = list()
    for n in range(N):
        k = np.random.choice(len(pi), p=pi)
        sample = np.random.normal(loc=mu[k], scale=sigma[k])
        data.append(sample)
    return data


def main():
    data = np.genfromtxt('./bdims.csv', delimiter=',', skip_header=1)  # [:,-2]
    data = data[:, -3]
    N = data.shape[0]
    K = 2  # two components GMM
    tot_iterations = 100  # stopping criteria

    # Step-1 (Init)
    mu = np.random.uniform(low=42.0, high=95.0, size=K)
    sigma = np.random.uniform(low=5.0, high=10.0, size=K)
    pi = np.ones(K) * (1.0 / K)  # mixing coefficients
    r = np.zeros([K, N])  # responsibilities
    nll_list = list()  # store the neg log-likelihood

    for iteration in range(tot_iterations):
        # Step-2 (E-Step)
        for k in range(K):
            r[k, :] = pi[k] * norm.pdf(x=data, loc=mu[k], scale=sigma[k])
        r = r / np.sum(r, axis=0)  # [K,N] -> [N]

        # Step-3 (M-Step)
        N_k = np.sum(r, axis=1)  # [K,N] -> [K]
        for k in range(K):
            # update means
            mu[k] = np.sum(r[k, :] * data) / N_k[k]
            # update variances
            numerator = r[k] * (data - mu[k]) ** 2
            sigma[k] = np.sqrt(np.sum(numerator) / N_k[k])
            # update weights
        pi = N_k / N

        likelihood = 0.0
        for k in range(K):
            likelihood += pi[k] * norm.pdf(x=data, loc=mu[k], scale=sigma[k])
        nll_list.append(-np.sum(np.log(likelihood)))
        # Check for invalid negative log-likelihood (NLL)
        # The NLL is invalid if NLL_t-1 < NLL_t
        # Note that this can happen for round-off errors.
        if (len(nll_list) >= 2):
            if (nll_list[-2] < nll_list[-1]): raise Exception("[ERROR] invalid NLL: " + str(nll_list[-2:]))

        print("Iteration: " + str(iteration) + "; NLL: " + str(nll_list[-1]))
        print("Mean " + str(mu) + "\nStd " + str(sigma) + "\nWeights " + str(pi) + "\n")

        # Step-4 (Check)
        if (iteration == tot_iterations - 1): break  # check iteration

    plot_likelihood(nll_list)
    data_gmm = sampler(pi, mu, sigma, N=1000)
    plot_distributions(data, data_gmm, mu, sigma, K, color="green", color_sampled="red", name="plot_sampler.png")


if __name__ == "__main__":
    main()