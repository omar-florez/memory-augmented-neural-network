import numpy as np
import ipdb

#-----------------------------------------------------------------------------------------------------------------------
#statistical functions

#the lower the better. For example, log of P(1.0) is 0.0
def entropy(x_continous):
    x_continous = x_continous[np.nonzero(x_continous)]
    c_normalized = get_probabilities(x_continous)
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    h = -sum(c_normalized * np.log(c_normalized))
    return h

def entropy_normalized(x_continous):
    x_continous = x_continous[np.nonzero(x_continous)]
    #corner case: all zeros vector or only one element. Entropy is zero.
    if len(x_continous) < 2:
        return 0.0
    c_normalized = get_probabilities(x_continous)
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    h = -sum(c_normalized * np.log(c_normalized))/np.log(len(x_continous))
    return h

def histogram(x_continous):
    x_discrete = np.histogram(x_continous, bins=10, range=(0, 1.0), density=True)[0]
    return x_discrete

def get_probabilities(x_continous):
    x_normalized = np.array(x_continous) / float(np.sum(x_continous))
    return x_normalized

def get_histogram(x_continous, number_bins=10):
    x_discrete = np.histogram([x for x in x_continous if x != 0.0], bins=number_bins, range=(0, 1.0), density=True)[0]
    c_normalized = x_discrete / float(np.sum(x_discrete))
    return c_normalized

#The lower the better. Idead value of this loss function: 0.0
#1.0*log(1,0) = 0.0
def crossentropy_masked(x_continous):
    x_normalized = get_probabilities(x_continous)
    x_normalized += 1e-4

    canonical_mask =  np.zeros(len(x_continous))
    canonical_mask[0] = 1.0
    canonical_mask += 1e-4
    canonical_mask = canonical_mask / float(np.sum(canonical_mask))

    ch = -sum([np.log(x)*y for (x,y) in zip(x_normalized, canonical_mask)])
    return ch

def crossentropy_masked_normalized(x_continous):
    x_normalized = get_probabilities(x_continous)
    x_normalized += 1e-4

    canonical_mask =  np.zeros(len(x_continous))
    canonical_mask[0] = 1.0
    canonical_mask += 1e-4
    canonical_mask = canonical_mask / float(np.sum(canonical_mask))

    ch = -sum([np.log(x)*y for (x,y) in zip(x_normalized, canonical_mask)])
    return ch/np.log(len(x_continous))


def L2_ranking_masked(x_continous):
    x_normalized = get_probabilities(x_continous)
    #1.0 is perfect similarity and is assigned to first NN in target distribution
    target_distribution = np.zeros(len(x_continous))
    target_distribution[0] = 1.0
    loss = np.sqrt(np.square(target_distribution-np.array(x_normalized)).sum())
    return loss

def L1_ranking_masked(x_continous):
    x_normalized = get_probabilities(x_continous)
    #1.0 is perfect similarity and is assigned to first NN in target distribution
    target_distribution = np.zeros(len(x_continous))
    target_distribution[0] = 1.0
    loss = np.dot(target_distribution, x_normalized)
    return loss

if __name__ == '__main__':

    print('---------------------------------------------------------------')
    x_countinous = [0.73792613, 0.25566521, 0.21475521, 0., 0., 0., 0., 0., 0., 0.]
    x_countinous = np.array(x_countinous)
    print('x: ', x_countinous);
    print('get_probabilities: ', get_probabilities(x_countinous))
    print('entropy: ', entropy(x_countinous))
    print('entropy_normalized: ', entropy_normalized(x_countinous))
    print('crossentropy_masked: ', crossentropy_masked(x_countinous))
    print('crossentropy_masked_average: ', crossentropy_masked_normalized(x_countinous))
    print('L1_ranking_masked: ', L1_ranking_masked(x_countinous))
    print('L2_ranking_masked: ', L2_ranking_masked(x_countinous))

    print('---------------------------------------------------------------')
    x_countinous = [0.99, 0.001, 0.001, 0., 0., 0., 0., 0., 0., 0.]
    x_countinous = np.array(x_countinous)
    print('x: ', x_countinous);
    print('get_probabilities: ', get_probabilities(x_countinous))
    print('entropy: ', entropy(x_countinous))
    print('entropy_normalized: ', entropy_normalized(x_countinous))
    print('crossentropy_masked: ', crossentropy_masked(x_countinous))
    print('crossentropy_masked_average: ', crossentropy_masked_normalized(x_countinous))
    print('L1_ranking_masked: ', L1_ranking_masked(x_countinous))
    print('L2_ranking_masked: ', L2_ranking_masked(x_countinous))

    print('---------------------------------------------------------------')
    x_countinous = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    x_countinous = np.array(x_countinous)
    print('x: ', x_countinous);
    print('get_probabilities: ', get_probabilities(x_countinous))
    print('entropy: ', entropy(x_countinous))
    print('entropy_normalized: ', entropy_normalized(x_countinous))
    print('crossentropy_masked: ', crossentropy_masked(x_countinous))
    print('crossentropy_masked_average: ', crossentropy_masked_normalized(x_countinous))
    print('L1_ranking_masked: ', L1_ranking_masked(x_countinous))
    print('L2_ranking_masked: ', L2_ranking_masked(x_countinous))

    print('---------------------------------------------------------------')
    x_countinous = [0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    x_countinous = np.array(x_countinous)
    print('x: ', x_countinous);
    print('get_probabilities: ', get_probabilities(x_countinous))
    print('entropy: ', entropy(x_countinous))
    print('entropy_normalized: ', entropy_normalized(x_countinous))
    print('crossentropy_masked: ', crossentropy_masked(x_countinous))
    print('crossentropy_masked_average: ', crossentropy_masked_normalized(x_countinous))
    print('L1_ranking_masked: ', L1_ranking_masked(x_countinous))
    print('L2_ranking_masked: ', L2_ranking_masked(x_countinous))
