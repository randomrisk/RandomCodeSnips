# This script is to calculate commonly used features
import numpy as np
from scipy.signal import lfilter
from scipy.signal import welch
from spectrum import arburg

def arithmetic_mean(X):
    return np.mean(X)

def autoregressive_model(X, order=4):
    return arburg(X, order)[0][1:]

def bandpower(X, fs, band): 
    f, Pxx = welch(X, fs=fs)
    idx = np.logical_and(f >= band[0], f <= band[1])
    return np.trapz(Pxx[idx], f[idx])

def bandpower_delta(X, fs):
    return bandpower(X, fs, [1, 4])

def bandpower_theta(X, fs):
    return bandpower(X, fs, [4, 8])

def bandpower_alpha(X, fs):
    return bandpower(X, fs, [8, 12])

def bandpower_beta(X, fs):
    return bandpower(X, fs, [12, 30])

def bandpower_gamma(X, fs):
    return bandpower(X, fs, [30, 64])

def first_difference(X):
    return np.mean(np.abs(np.diff(X)))

def hjorth_activity(X):
    return np.var(X)

def hjorth_mobility(X):
    d1 = np.diff(X)
    return np.std(d1) / np.std(X)

def hjorth_complexity(X):
    d1 = np.diff(X)
    d2 = np.diff(d1)
    return (np.std(d2) / np.std(d1)) / hjorth_mobility(X)

def kurtosis(X):
    return scipy.stats.kurtosis(X)

def log_energy_entropy(X):
    return np.sum(np.log(X**2))

def log_root_sum_sequential_variation(X):
    return np.log10(np.sqrt(np.sum(np.diff(X)**2)))

def maximum(X):
    return np.max(X)

def mean_curve_length(X):
    return np.mean(np.abs(np.diff(X)))

def mean_energy(X):
    return np.mean(X**2)

def mean_teager_energy(X):
    return np.mean(X[1:-1]**2 - X[2:] * X[:-2])

def median(X):
    return np.median(X)

def minimum(X):
    return np.min(X)

def normalized_first_difference(X):
    return first_difference(X) / np.std(X)

def normalized_second_difference(X):
    return np.mean(np.abs(np.diff(X, n=2))) / np.std(X)

def ratio_bandpower_alpha_beta(X, fs):
    return bandpower_beta(X, fs) / bandpower_alpha(X, fs)

def renyi_entropy(X, alpha=2):
    p = X**2 / np.sum(X**2)
    return (1 / (1 - alpha)) * np.log2(np.sum(p**alpha))

def second_difference(X):
    return np.mean(np.abs(np.diff(X, n=2)))

def shannon_entropy(X):
    p = X**2 / np.sum(X**2)
    return -np.sum(p * np.log2(p))

def skewness(X):
    return scipy.stats.skew(X)

def standard_deviation(X):
    return np.std(X)

def tsallis_entropy(X, alpha=2):
    p = X**2 / np.sum(X**2)
    return (1 / (alpha - 1)) * (1 - np.sum(p**alpha))

def variance(X):
    return np.var(X)

def extract_features(X, feature_type, **kwargs):
    features = {
        'mcl': mean_curve_length,
        'ha': hjorth_activity,
        'hm': hjorth_mobility,
        'hc': hjorth_complexity,
        '1d': first_difference,
        'n1d': normalized_first_difference,
        '2d': second_difference,
        'n2d': normalized_second_difference,
        'me': mean_energy,
        'mte': mean_teager_energy,
        'lrssv': log_root_sum_sequential_variation,
        'te': tsallis_entropy,
        'sh': shannon_entropy,
        'le': log_energy_entropy,
        're': renyi_entropy,
        'am': arithmetic_mean,
        'sd': standard_deviation,
        'var': variance,
        'md': median,
        'max': maximum,
        'min': minimum,
        'ar': autoregressive_model,
        'kurt': kurtosis,
        'skew': skewness,
        'bpd': bandpower_delta,
        'bpt': bandpower_theta,
        'bpa': bandpower_alpha,
        'bpb': bandpower_beta,
        'bpg': bandpower_gamma,
        'rba': ratio_bandpower_alpha_beta,
    }
    return features[feature_type](X, **kwargs)