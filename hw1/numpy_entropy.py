#!/usr/bin/env python3
import argparse
import numpy as np

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", help="Path to data file, containing data points of our dataset")
parser.add_argument("--model_path", help="Path to the model file, containing its probability distribution")

def main(args: argparse.Namespace) -> tuple[float, float, float]:
    # 1. load data and compute entropy of the data!
    dataset, px = load_data(args.data_path)
    entropy = H(px)

    # 2. load model
    model = load_model(args.model_path, dataset)
    # do not show runtime warning for log(0) = inf
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_entropy = cross_H(px, model)
        kl_divergence = diverg_KL(px, model)

        print(f"Entropy: {entropy:.2f} nats")
        print(f"Crossentropy: {cross_entropy:.2f} nats")
        print(f"KL divergence: {kl_divergence:.2f} nats")

    return entropy, cross_entropy, kl_divergence

def H(data : np.ndarray):
    """
    Computes the entropy of a given data distribution, which is 
    measure of the uncertainty in a probability distribution.

    --> H(P) = -∑ P(x) log P(x), where P(x) is the probability of each data point.

    Parameters
    ----------
    data_dist : dict
        dictionary of values and the respective array of probability values

    Returns
    -------
    entropy : float
        numeric value of entropy
    """
    entropy = -float(np.sum(data * np.log(data)))
    return entropy


def cross_H(data : np.ndarray, model : np.ndarray):
    """
    Computes the cross-entropy, which measures the average number of bits needed to encode events
    from the true distribution when using an estimated model distribution. It is computed as:

    --> H(P, Q) = -∑ P(x) log Q(x), where P(x) is the true distribution and Q(x) the estimated one.

    Parameters
    ----------
    data : numpy.ndarray
        real probability distribution values
    model : numpy.ndarray
        model, estimated probability distribution values

    Returns
    -------
    cross_entropy : float
        The cross-entropy value between the data and model distributions.
    """
    cross_entropy = - float(np.sum(data * np.log(model)))
    return cross_entropy


def diverg_KL(data : np.ndarray, model : np.ndarray):
    """
    Computes the Kullback-Leibler (KL) divergence between the data distribution and the model distribution,
    measuring how they differ from one another.

    --> D_kl(P || Q) = ∑ P(x) (log P(x) - log Q(x)),
    where P(x) is the true probability distribution and Q(x) is the model probability.

    Parameters
    ----------
    data : numpy.ndarray
        real probability distribution values
    model : numpy.ndarray
        model, estimated probability distribution values

    Returns
    -------
    kl : float
        computed KL divergence value
    """
    kl = float(np.sum(data * (np.log(data) - np.log(model))))
    return kl

##
## --- auxiliary functions
##

def load_data(data_path):
    """
    Loads textual data from a file whose lines consist of data points of a dataset.
    Then, it estimates the frequency-based probabilities of each one the data points.

    Parameters
    ----------
    data_path : str
        file path/name of the dataset file
    
    Returns
    -------
    data_points : numpy.ndarray
        array of datapoints found in a dataset
    probs : numpy.ndarray
        datapoints' respective frequency-based probabilities
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.read().splitlines()
    
    data_points, counts = np.unique(data, return_counts=True)
    data_probs = counts / counts.sum()
    return data_points, data_probs

def load_model(model_path, dataset):
    """
    Loads estimated probability distribution of a model, and aligns it
    with real probability data by setting unseen datapoints to 0.

    Parameters
    ----------
    model_path : str
        file path/name of the model's probability distribution
    dataset : numpy.ndarray
        datapoints found in a dataset
    Returns
    -------
    model : numpy.ndarray
        model's probability distributions for all values in data
    """
    dist = {}
    with open(model_path, "r", encoding="utf-8") as f:
        for line in f:
            key, prob = line.strip().split("\t")
            dist[key] = float(prob)
    
    model = np.array([dist.get(p, 0) for p in dataset])
    return model


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    entropy, cross_entropy, kl_divergence = main(main_args)

    print(f"Total entropy: {entropy:.2f}")
    print(f"Total cross-entropy: {cross_entropy:.2f}")
    print(f"Total KL divergence: {kl_divergence:.2f}")
