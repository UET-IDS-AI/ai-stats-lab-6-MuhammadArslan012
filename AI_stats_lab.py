import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    # Convert to numpy array
    data = np.array(data)

    # ✅ Validation
    if data.size == 0:
        raise ValueError("Data must not be empty")

    if not (0 < theta < 1):
        raise ValueError("Theta must be in (0,1)")

    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must contain only 0 and 1")

    # ✅ Log-Likelihood
    log_likelihood = np.sum(
        data * np.log(theta) + (1 - data) * np.log(1 - theta)
    )

    return float(log_likelihood)


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    data = np.array(data)

    # ✅ Validation
    if data.size == 0:
        raise ValueError("Data must not be empty")

    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must contain only 0 and 1")

    # Default candidates
    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    # ✅ Count
    num_successes = int(np.sum(data))
    num_failures = int(len(data) - num_successes)

    # ✅ MLE (mean of data)
    mle = num_successes / len(data)

    # ✅ Log-likelihoods
    log_likelihoods = {}
    for theta in candidate_thetas:
        try:
            ll = bernoulli_log_likelihood(data, theta)
            log_likelihoods[theta] = ll
        except ValueError:
            log_likelihoods[theta] = float('-inf')

    # ✅ Best candidate
    best_candidate = None
    best_value = float('-inf')

    for theta in candidate_thetas:
        if log_likelihoods[theta] > best_value:
            best_value = log_likelihoods[theta]
            best_candidate = theta

    return {
        'mle': mle,
        'num_successes': num_successes,
        'num_failures': num_failures,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }


def poisson_log_likelihood(data, lam):
    data = np.array(data)

    # ✅ Validation
    if data.size == 0:
        raise ValueError("Data must not be empty")

    if lam <= 0:
        raise ValueError("Lambda must be > 0")

    if not np.all((data >= 0) & (data == data.astype(int))):
        raise ValueError("Data must be nonnegative integers")

    # ✅ Log-Likelihood
    log_likelihood = 0.0
    for x in data:
        log_likelihood += x * math.log(lam) - lam - math.lgamma(x + 1)

    return float(log_likelihood)


def poisson_mle_analysis(data, candidate_lambdas=None):
    data = np.array(data)

    # ✅ Validation
    if data.size == 0:
        raise ValueError("Data must not be empty")

    if not np.all((data >= 0) & (data == data.astype(int))):
        raise ValueError("Data must be nonnegative integers")

    # Default candidates
    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    n = len(data)
    total_count = int(np.sum(data))

    # ✅ MLE (mean)
    sample_mean = total_count / n
    mle = sample_mean

    # ✅ Log-likelihoods
    log_likelihoods = {}
    for lam in candidate_lambdas:
        try:
            ll = poisson_log_likelihood(data, lam)
            log_likelihoods[lam] = ll
        except ValueError:
            log_likelihoods[lam] = float('-inf')

    # ✅ Best candidate
    best_candidate = None
    best_value = float('-inf')

    for lam in candidate_lambdas:
        if log_likelihoods[lam] > best_value:
            best_value = log_likelihoods[lam]
            best_candidate = lam

    return {
        'mle': mle,
        'sample_mean': sample_mean,
        'total_count': total_count,
        'n': n,
        'log_likelihoods': log_likelihoods,
        'best_candidate': best_candidate
    }
