import time
import argparse
import os
import warnings
import numpy as np
from matplotlib import pyplot as plt


def get_moments_estimates_1(ordered_data):
    """
    Function to calculate first moments array given an ordered data
    sequence. Decreasing ordering is required.
    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st moment (Hill estimator)
                      is calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
    """
    logs_1 = np.log(ordered_data)
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    return M1

def get_moments_estimates_2(ordered_data):
    """
    Function to calculate first and second moments arrays
    given an ordered data sequence. 
    Decreasing ordering is required.
    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator) and 2nd moments 
                      are calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
        M2: numpy array of 2nd moments corresponding to all 
            possible order statistics of the dataset.
    """
    logs_1 = np.log(ordered_data)
    logs_2 = (np.log(ordered_data))**2
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    M2 = (1./k_vector)*logs_2_cumsum - (2.*logs_1[1:]/k_vector)*logs_1_cumsum\
         + logs_2[1:]
    return M1, M2

def get_moments_estimates_3(ordered_data):
    """
    Function to calculate first, second and third moments 
    arrays given an ordered data sequence. 
    Decreasing ordering is required.
    Args:
        ordered_data: numpy array of ordered data for which
                      the 1st (Hill estimator), 2nd and 3rd moments 
                      are calculated.
    Returns:
        M1: numpy array of 1st moments (Hill estimator)
            corresponding to all possible order statistics
            of the dataset.
        M2: numpy array of 2nd moments corresponding to all 
            possible order statistics of the dataset.
        M3: numpy array of 3rd moments corresponding to all 
            possible order statistics of the dataset.
    """
    logs_1 = np.log(ordered_data)
    logs_2 = (np.log(ordered_data))**2
    logs_3 = (np.log(ordered_data))**3
    logs_1_cumsum = np.cumsum(logs_1[:-1])
    logs_2_cumsum = np.cumsum(logs_2[:-1])
    logs_3_cumsum = np.cumsum(logs_3[:-1])
    k_vector = np.arange(1, len(ordered_data))
    M1 = (1./k_vector)*logs_1_cumsum - logs_1[1:]
    M2 = (1./k_vector)*logs_2_cumsum - (2.*logs_1[1:]/k_vector)*logs_1_cumsum\
         + logs_2[1:]
    M3 = (1./k_vector)*logs_3_cumsum - (3.*logs_1[1:]/k_vector)*logs_2_cumsum\
         + (3.*logs_2[1:]/k_vector)*logs_1_cumsum - logs_3[1:]
    # cleaning exceptional cases
    clean_indices = np.where((M2 <= 0) | (M3 == 0) | (np.abs(1.-(M1**2)/M2) < 1e-10)\
                             |(np.abs(1.-(M1*M2)/M3) < 1e-10))
    M1[clean_indices] = np.nan
    M2[clean_indices] = np.nan
    M3[clean_indices] = np.nan
    return M1, M2, M3

def hill_dbs(ordered_data, t_bootstrap=0.5,
             r_bootstrap=500, eps_stop=1.0,
             verbose=False, diagn_plots=False):
    """
        Function to perform double-bootstrap procedure for
        Hill estimator.
        Args:
            ordered_data: numpy array for which double-bootstrap
                          is performed. Decreasing ordering is required.
            t_bootstrap:  parameter controlling the size of the 2nd
                          bootstrap. Defined from n2 = n*(t_bootstrap).
            r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                          bootstraps.
            eps_stop:     parameter controlling range of AMSE minimization.
                          Defined as the fraction of order statistics to consider
                          during the AMSE minimization step.
            verbose:      flag controlling bootstrap verbosity.
            diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                          plots.
        Returns:
            k_star:     number of order statistics optimal for estimation
                        according to the double-bootstrap procedure.
            x1_arr:     array of fractions of order statistics used for the
                        1st bootstrap sample.
            n1_amse:    array of AMSE values produced by the 1st bootstrap
                        sample.
            k1_min:     value of fraction of order statistics corresponding
                        to the minimum of AMSE for the 1st bootstrap sample.
            max_index1: index of the 1st bootstrap sample's order statistics
                        array corresponding to the minimization boundary set
                        by eps_stop parameter.
            x2_arr:     array of fractions of order statistics used for the
                        2nd bootstrap sample.
            n2_amse:    array of AMSE values produced by the 2nd bootstrap
                        sample.
            k2_min:     value of fraction of order statistics corresponding
                        to the minimum of AMSE for the 2nd bootstrap sample.
            max_index2: index of the 2nd bootstrap sample's order statistics
                        array corresponding to the minimization boundary set
                        by eps_stop parameter.
    """
    if verbose:
        print
        "Performing Hill double-bootstrap..."
    n = len(ordered_data)
    eps_bootstrap = 0.5 * (1 + np.log(int(t_bootstrap * n)) / np.log(n))
    n1 = int(n ** eps_bootstrap)
    samples_n1 = np.zeros(n1 - 1)
    good_counts1 = np.zeros(n1 - 1)
    k1 = None
    k2 = None
    min_index1 = 1
    min_index2 = 1
    while k2 == None:
        # first bootstrap with n1 sample size
        for i in range(r_bootstrap):
            sample = np.random.choice(ordered_data, n1, replace=True)
            sample[::-1].sort()#从大到小排列
            M1, M2 = get_moments_estimates_2(sample)
            current_amse1 = (M2 - 2. * (M1) ** 2) ** 2
            samples_n1 += current_amse1
            good_counts1[np.where(current_amse1 != np.nan)] += 1
        averaged_delta = samples_n1 / good_counts1

        max_index1 = (np.abs(np.linspace(1. / n1, 1.0, n1) - eps_stop)).argmin()
        k1 = np.nanargmin(averaged_delta[min_index1:max_index1]) + 1 + min_index1  # take care of indexing
        if diagn_plots:
            n1_amse = averaged_delta
            x1_arr = np.linspace(1. / n1, 1.0, n1)

        # second bootstrap with n2 sample size
        n2 = int(n1 * n1 / float(n))
        samples_n2 = np.zeros(n2 - 1)
        good_counts2 = np.zeros(n2 - 1)

        for i in range(r_bootstrap):
            sample = np.random.choice(ordered_data, n2, replace=True)
            sample[::-1].sort()
            M1, M2 = get_moments_estimates_2(sample)
            current_amse2 = (M2 - 2. * (M1 ** 2)) ** 2
            samples_n2 += current_amse2
            good_counts2[np.where(current_amse2 != np.nan)] += 1
        max_index2 = (np.abs(np.linspace(1. / n2, 1.0, n2) - eps_stop)).argmin()
        averaged_delta = samples_n2 / good_counts2

        k2 = np.nanargmin(averaged_delta[min_index2:max_index2]) + 1 + min_index2  # take care of indexing
        if diagn_plots:
            n2_amse = averaged_delta
            x2_arr = np.linspace(1. / n2, 1.0, n2)

        if k2 > k1:
            print
            "Warning (Hill): k2 > k1, AMSE false minimum suspected, resampling..."
            # move left AMSE boundary to avoid numerical issues
            min_index1 = min_index1 + int(0.005 * n)
            min_index2 = min_index2 + int(0.005 * n)
            k2 = None

    '''
    # this constant is provided in the Danielsson's paper
    # use instead of rho below if needed
    rho = (np.log(k1)/(2.*np.log(n1) - np.log(k1)))\
          **(2.*(np.log(n1) - np.log(k1))/(np.log(n1)))
    '''

    # this constant is provided in Qi's paper
    rho = (1. - (2 * (np.log(k1) - np.log(n1)) / (np.log(k1)))) ** (np.log(k1) / np.log(n1) - 1.)

    k_star = (k1 * k1 / float(k2)) * rho
    k_star = int(np.round(k_star))

    # enforce k_star to pick 2nd value (rare cases of extreme cutoffs)
    if k_star == 0:
        k_star = 2
    if int(k_star) >= len(ordered_data):
        print
        "WARNING: estimated threshold k is larger than the size of data"
        k_star = len(ordered_data) - 1
    if verbose:
        print
        "--- Hill double-bootstrap information ---"
        print
        "Size of the 1st bootstrap sample n1:", n1
        print
        "Size of the 2nd bootstrap sample n2:", n2
        print
        "Estimated k1:", k1
        print
        "Estimated k2:", k2
        print
        "Estimated constant rho:", rho
        print
        "Estimated optimal k:", k_star
        print
        "-----------------------------------------"
    if not diagn_plots:
        x1_arr, x2_arr, n1_amse, n2_amse = None, None, None, None
    return k_star, x1_arr, n1_amse, k1/float(n1), max_index1, x2_arr, n2_amse, k2 / float(n2), max_index2


def hill_estimator(ordered_data,
                   bootstrap = True, t_bootstrap = 0.5,
                   r_bootstrap = 50, verbose = False,
                   diagn_plots = False, eps_stop = 0.99):
    """
    Function to calculate Hill estimator for a given dataset.
    If bootstrap flag is True, double-bootstrap procedure
    for estimation of the optimal number of order statistics is
    performed.
    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        bootstrap:    flag to switch on/off double-bootstrap procedure.
        t_bootstrap:  parameter controlling the size of the 2nd
                      bootstrap. Defined from n2 = n*(t_bootstrap).
        r_bootstrap:  number of bootstrap resamplings for the 1st and 2nd
                      bootstraps.
        eps_stop:     parameter controlling range of AMSE minimization.
                      Defined as the fraction of order statistics to consider
                      during the AMSE minimization step.
        verbose:      flag controlling bootstrap verbosity. 
        diagn_plots:  flag to switch on/off generation of AMSE diagnostic
                      plots.
    Returns:
        results: list containing an array of order statistics,
                 an array of corresponding tail index estimates,
                 the optimal order statistic estimated by double-
                 bootstrap and the corresponding tail index,
                 an array of fractions of order statistics used for
                 the 1st bootstrap sample with an array of corresponding
                 AMSE values, value of fraction of order statistics
                 corresponding to the minimum of AMSE for the 1st bootstrap
                 sample, index of the 1st bootstrap sample's order statistics
                 array corresponding to the minimization boundary set
                 by eps_stop parameter; and the same characteristics for the
                 2nd bootstrap sample.
    """
    k_arr = np.arange(1, len(ordered_data))
    xi_arr = get_moments_estimates_1(ordered_data)
    if bootstrap:
        results = hill_dbs(ordered_data,
                           t_bootstrap = t_bootstrap,
                           r_bootstrap = r_bootstrap,
                           verbose = verbose, 
                           diagn_plots = diagn_plots,
                           eps_stop = eps_stop)
        k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2 = results
        while k_star == None:
            print("Resampling...")
            results = hill_dbs(ordered_data,
                           t_bootstrap = t_bootstrap,
                           r_bootstrap = r_bootstrap,
                           verbose = verbose, 
                           diagn_plots = diagn_plots,
                           eps_stop = eps_stop)
            k_star, x1_arr, n1_amse, k1, max_index1, x2_arr, n2_amse, k2, max_index2 = results
        xi_star = xi_arr[k_star-1]
        #print("Adjusted Hill estimated gamma:", 1 + 1./xi_star)
        #print("**********")
    else:
        k_star, xi_star = None, None
        x1_arr, n1_amse, k1, max_index1 = 4*[None]
        x2_arr, n2_amse, k2, max_index2 = 4*[None]
    results = [k_arr, xi_arr, k_star, xi_star, x1_arr, n1_amse, k1, max_index1,\
               x2_arr, n2_amse, k2, max_index2]
    return results

def smooth_hill_estimator(ordered_data, r_smooth = 2):
    """
    Function to calculate smooth Hill estimator for a
    given ordered dataset.
    Args:
        ordered_data: numpy array for which tail index estimation
                      is performed. Decreasing ordering is required.
        r_smooth:     integer parameter controlling the width
                      of smoothing window. Typically small
                      value such as 2 or 3.
    Returns:
        k_arr:  numpy array of order statistics based on the data provided.
        xi_arr: numpy array of tail index estimates corresponding to 
                the order statistics array k_arr.
    """
    n = len(ordered_data)
    M1 = get_moments_estimates_1(ordered_data)
    xi_arr = np.zeros(int(np.floor(float(n)/r_smooth)))
    k_arr = np.arange(1, int(np.floor(float(n)/r_smooth))+1)
    xi_arr[0] = M1[0]
    bin_lengths = np.array([1.]+[float((r_smooth-1)*k) for k in k_arr[:-1]])
    cum_sum = 0.0
    for i in range(1, r_smooth*int(np.floor(float(n)/r_smooth))-1):
        k = i
        cum_sum += M1[k]
        if (k+1) % (r_smooth) == 0:
            xi_arr[int(k+1)/int(r_smooth)] = cum_sum
            cum_sum -= M1[int(k+1)/int(r_smooth)]
    xi_arr = xi_arr/bin_lengths
    return k_arr, xi_arr
