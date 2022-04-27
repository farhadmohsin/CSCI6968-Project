import numpy as np
from voting_utils import *
import pandas as pd
    
#%% sampling algorithm

def borda_sample(votes, ns):
    '''
    Parameters
    ----------
    votes : preference profile of single group.
    ns : sampling parameter

    Returns
    -------
    S_sample : noisy alternative scores for samples
    '''
    n, m = votes.shape
    S_sample = np.zeros(m)
    
    # for each alternative
    samples = 0
    for j in range(m):
        # get random number of samples using binom(ns, 2/m)
        # this gives O(ns) pairwise samples
        n_x = np.random.binomial(ns, 2/m)
        samples += n_x
        for i in range(n_x):
            
            # sample an alternative other than j
            k = np.random.randint(m)
            while k==j:
                k = np.random.randint(m)
            
            # sample any voter
            i = np.random.randint(n)
            vote = votes[i]
            
            # if voter i has j \succ k, then S_sample[j] += 1
            if rank_j(vote, j) < rank_j(vote, k):
                S_sample[j] += 1
    print(n, samples, np.sum(S_sample))
    return np.argwhere(S_sample == np.max(S_sample)), S_sample

def random_fair_borda(votes1, votes2, ns, gamma):
    '''
    Function for fair version of Borda using sampling
    
    Parameters
        votes1: group 1 preference profile
        votes2: group 2 preference profile
        ns: sampling parameter, determines eps
        gamma: randomization parameter, determined by delta
    '''
    
    n1, m = votes1.shape
    n2, m = votes2. shape
    
    # with probability gamma, return a random alternative
    u = np.random.random()
    if u < gamma:
        return np.random.randint(m)
    
    # otherwise, with probability (1-gamma), apply the fair sampling method
    
    # apply sampling on each group's preference profile
    _, score1 = borda_sample(votes1, ns)
    _, score2 = borda_sample(votes2, ns)
    print('done')
    
    # calculate imbalance and randomly return one of the most fair alternatives
    score_diff = np.abs(score1 - score2)
    fair_j = np.argwhere(score_diff == np.min(score_diff))
    return np.random.choice(fair_j[0])

# %% Laplac DP mechanism

def add_laplace_noise(scores, e, n):
    '''
    Parameters
    ----------
    scores : score for each alternative.
    e : e is epsilon.
    n : n is number of voters, needed to normalize score

    Returns
    -------
    returns score with added Laplace noise

    '''
    # score is normalized by n
    m = len(scores)
    # calculate and apply laplace noise
    loc, scale = 0, m * (m-1)/(2 * n *e)
    noise = np.random.laplace(loc, scale, size = m)
    
    return np.add(scores, noise)

def laplace_fair_Borda(votes1, votes2, e):
    n1, m = votes1.shape
    n2, m = votes2.shape
    
    # get actual scores for each alternative
    _, s1 = Borda_winner(votes1)
    _, s2 = Borda_winner(votes2)
    
    # normalize scores by number of voters
    s1 = s1/n1
    s2 = s2/n2
    
    # add Laplace nosie to scores for both groups
    noisy_s1 = add_laplace_noise(s1, e, n1)
    noisy_s2 = add_laplace_noise(s2, e, n2)
    
    
    # calculate imbalance and randomly return one of the most fair alternatives
    noisy_diff = np.abs(noisy_s1 - noisy_s2)
    fair_j = np.argwhere(noisy_diff == np.min(noisy_diff))
    
    return np.random.choice(fair_j[0])

# %% define fairness

def util_diff(votes1, votes2):
    '''
    Parameters
    ----------
    votes1 : group 1 pref. profile
    votes2 : group 2 pref. profile

    Returns
    -------
    imabalnce for all alternatives

    '''
    n1, m = votes1.shape
    n2, m = votes1.shape
    
    _, s1 = Borda_winner(votes1)
    _, s2 = Borda_winner(votes2)
    
    s1 = s1/n1
    s2 = s2/n2
    
    return np.abs(s1 - s2)

def joint_score(votes1, votes2):
    '''
    Parameters
    ----------
    votes1 : group 1 pref profile
    votes2 : group 2 pref profile

    Returns
    -------
    returns average utility for each alternative (considers whole population)
    '''
    n1, m = votes1.shape
    n2, m = votes1.shape
    
    _, s1 = Borda_winner(votes1)
    _, s2 = Borda_winner(votes2)
    
    s = np.add(s1, s2)
    
    return s / (n1 + n2)


if __name__ == '__main__':
    # %% Settting
    n2_range = np.array([500, 1000, 2000])
    n1_range = n2_range * 2
    
    m = 4
    
    # %%
    delta = 0.01
    eps_all = [0.3, 0.6, 1.0]
    ns_all = [np.math.ceil(2*m/(eps**2) * np.math.log(m / delta)) for eps in eps_all]
    
    for n1, n2 in zip(n1_range, n2_range):
        for eps, ns in zip(eps_all, ns_all):
            print(f'{n1=},{n2=},{ns=},{eps=}')
    
    # %%
    
    # We will geneate random votes using Mallow's distributions
    
    # Mallow's distribution parameters
    W1 = np.array([0, 1, 2, 3])
    W2 = np.array([1, 2, 0, 3])
    phi = 0.6
    
    no_samples = 100
    
    results = []
    
    diff_bounds = []
    util_bounds = []
    
    for n1, n2 in zip(n1_range, n2_range):
        # print(n1, n2)
        
        # _diffs and _utils are there to compare the highest/lowest values as baselines
        min_diffs = []
        max_diffs = []
        
        min_utils = []
        max_utils = []
        
        for idx in range(no_samples):
            # generate random preference profiles
            votes1 = gen_mallows_profile(n1, W1, phi)
            votes2 = gen_mallows_profile(n2, W2, phi)
            
            diff = util_diff(votes1, votes2)
            util = joint_score(votes1, votes2)
            min_diffs.append(np.min(diff))
            max_diffs.append(np.max(diff))
            
            min_utils.append(np.min(util))
            max_utils.append(np.max(util))
            
            for eps, ns in zip(eps_all, ns_all):
                
                # print(n1, n2, eps, idx)
                
                # sampling first
                # print('sampling mechanism')
                gamma = min([delta/2, 1/m])
                w_r = random_fair_borda(votes1, votes2, ns, gamma)
                results.append([idx, n1, n2, 'sampling', eps, diff[w_r], util[w_r]])
            
                # then laplace
                # print('Laplace mechanism')    
                w_l = laplace_fair_Borda(votes1, votes2, eps)
                results.append([idx, n1, n2, 'laplace', eps, diff[w_l], util[w_l]])
        
        diff_bounds.append([n1, n2, np.mean(min_diffs), np.mean(max_diffs)])
        util_bounds.append([n1, n2, np.mean(min_utils), np.mean(max_utils)])
                
    
    # %%
    df = pd.DataFrame(results, columns = ['idx', 'n1', 'n2', 'method', 'eps', 'diff','util'])  
    print(df.groupby(['n1','n2','method','eps']).mean())  
    
