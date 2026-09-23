import numpy as np
import ndd
from scipy.special import factorial
from scipy.special import softmax
import multiprocessing as mp
from itertools import combinations_with_replacement
from itertools import product

### Functions for performing IB analyses on empirical data ###

# compute mutual information using NSB method (use for actual human choice behavior)
def mutual_inf_nsb(x,y,ks):
    """
    Calculate mutual information using NSB method
    """
    # ks is a list of two values giving the cardinality of the two variables (i.e. number of unique values in each variable)
    ar = np.column_stack((x,y))
    mi = ndd.mutual_information(ar,ks)
    return np.log2(np.e)*mi #ndd returns nats - multiply by log2(e) to convert to bits

# convert the bead and jar sequences into X and Y arrays for IB analyses
# this function and the one below ensure that X, R, and Y are properly aligned given the way the experiment was run and encoded
def getXY_beads(bead_seq,jar_seq,wsize,encodeX=False,ref_wsize=None):
    if ref_wsize is None:
        ref_wsize = wsize
    Y = jar_seq[ref_wsize-1:-1]
    if encodeX:
        aux_base = 2**np.arange(wsize)
        X = np.array([sum(aux_base*np.array(bead_seq[i:(i+wsize)])) for i in np.arange(ref_wsize-wsize,len(bead_seq)-wsize)])
    else:
        X = np.array([bead_seq[i:(i+wsize)] for i in np.arange(ref_wsize-wsize,len(bead_seq)-wsize)])
    return X.astype(int),Y.astype(int)

def getR_beads(choices,wsize):
    return choices[wsize:]

def Iry_boot_iter(args):
    Remp, Yemp, Rcard, Ycard, baseseed, iterseed = args
    rng = np.random.default_rng(baseseed+iterseed)
    bootinds = rng.choice(Remp.shape[0],Remp.shape[0])
    return mutual_inf_nsb(Remp[bootinds],Yemp[bootinds],[Rcard,Ycard])

def get_bootstrapped_Iry(Remp,Yemp,iters=1000,Rcard=2,Ycard=2,seed=234,N_threads=1):
    with mp.Pool(processes=N_threads) as pool:
        results = [pool.apply_async(Iry_boot_iter, args=((Remp, Yemp, Rcard, Ycard, seed, iterseed),)) for iterseed in range(iters)]
        Iry_boot = [res.get() for res in results]
    return np.array(Iry_boot)

def get_Iry_CI(R,Y,CI,boot_iters,Rcard=2,Ycard=2,seed=234,N_threads=1):
    Iry_boot = get_bootstrapped_Iry(R,Y,boot_iters,Rcard=Rcard,Ycard=Ycard,seed=seed,N_threads=N_threads)
    return np.percentile(Iry_boot,CI)

def Iry_cond_boot_iter(args):
    Xemp, Remp, Yemp, Rcard, Ycard, baseseed, iterseed = args
    rng = np.random.default_rng(baseseed+iterseed)
    Xunique = np.unique(Xemp, axis=0)
    Remp_boot = []
    Yemp_boot = []
    for X in Xunique:
        # print(X)
        xinds = np.all(Xemp == X, axis=1)
        Remp_x = Remp[xinds]
        Yemp_x = Yemp[xinds]
        bootinds = rng.choice(Remp_x.shape[0], Remp_x.shape[0])
        Remp_boot.append(Remp_x[bootinds])
        Yemp_boot.append(Yemp_x[bootinds])

    Remp_boot = np.concatenate(Remp_boot)
    Yemp_boot = np.concatenate(Yemp_boot)
    return mutual_inf_nsb(Remp_boot, Yemp_boot, [Rcard, Ycard])

def get_bootstrapped_Iry_cond(Xemp, Remp, Yemp, iters=1000, Rcard=2, Ycard=2, seed=234, N_threads=1):
    with mp.Pool(processes=N_threads) as pool:
        results = [pool.apply_async(Iry_cond_boot_iter, args=((Xemp, Remp, Yemp, Rcard, Ycard, seed, iterseed),)) for iterseed in range(iters)]
        Iry_boot = [res.get() for res in results]
    return np.array(Iry_boot)

def get_Iry_cond_CI(X, R, Y, CI, boot_iters, Rcard=2, Ycard=2, seed=234, N_threads=1):
    Iry_boot = get_bootstrapped_Iry_cond(X, R, Y, boot_iters, Rcard=Rcard, Ycard=Ycard, seed=seed, N_threads=N_threads)
    return np.percentile(Iry_boot, CI)


# def get_bootstrapped_Iry(R,Y,iters=1000,Rcard=2,Ycard=2,seed=234,pseed=0):
#     Iry_boot = []

#     rng = np.random.default_rng(seed+pseed)

#     for ii in range(iters):
#         bootinds = rng.choice(R.shape[0],R.shape[0])
#         Iry_boot.append(mutual_inf_nsb(R[bootinds],Y[bootinds],[Rcard,Ycard]))

#     return np.array(Iry_boot)

# # with mp.Pool(processes=N_threads) as pool:
# #         results = [pool.apply_async(solve_IB_mp, args=((beta_array[ib], p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib),)) for ib in range(N_b)]
# #         results = [res.get() for res in results]
# #     I_XR = [res[0] for res in results]
# #     I_RY = [res[1] for res in results]
# #     betas = [res[2] for res in results]
# #     H_Rs = [res[3] for res in results]

# def get_bootstrapped_Iry_mp(args):
#     R, Y, iters, Rcard, Ycard, seed, pseed = args
#     return get_bootstrapped_Iry(R, Y, iters, Rcard, Ycard, seed, pseed)

def classify_by_bound(Ixr_arr, Iry_arr, Iry_CI_low_arr, Iry_CI_up_arr, IB_bounds,bound_labels=None):
    bound_class_arr = np.zeros_like(Ixr_arr, dtype=int)
    bound_dist_arr = np.full_like(Ixr_arr, np.inf, dtype=float)
    for ii, bound in enumerate(IB_bounds):
        Iry_interp = np.interp(Ixr_arr,bound['I_XR'],bound['I_RY'])
        Iry_dist = np.abs(Iry_arr - Iry_interp)
        on_bound = (Iry_CI_low_arr <= Iry_interp) & (Iry_CI_up_arr >= Iry_interp) & (Iry_dist < bound_dist_arr)
        bound_dist_arr[on_bound] = Iry_dist[on_bound]
        bound_class_arr[on_bound] = ii + 1 # add 1 to the class index so that 0 can be reserved for "none of the bounds"
    if bound_labels is not None:
        bound_labels = ['none'] + bound_labels
        bound_class_arr = np.array([bound_labels[i] for i in bound_class_arr])
    return bound_class_arr

### Functions for computing IB bounds/curves ###
def compute_sumlog(A,B):
    B[B==0] = 1e-12
    return np.sum(A * np.log2(B))

# compute I(X;R) using distributions, standalone version
def compute_I_XR(p_RgX,p_XgY,p_Y):
    p_X = np.sum(p_XgY * p_Y, axis=1, keepdims=True)
    p_R = np.sum(p_RgX * p_X, axis=0, keepdims=True)
    return np.sum(p_RgX*p_X*np.log2(p_RgX/p_R))

# compute I(X;R) using distributions, version for use within IB algorithm (where p_X and p_R are already computed and passed in as arguments)
# def compute_I_XR_(p_RgX,p_X,p_R):
#     return np.sum(p_RgX*p_X*np.log2(p_RgX/p_R))

# compute I(R;Y) using distributions, standalone version
def compute_I_RY(p_RgX,p_XgY,p_Y):
    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_YgX = p_XY / p_X
    p_R = np.sum(p_RgX * p_X, axis=0, keepdims=True)
    p_XgR = (p_RgX * p_X) / p_R
    p_YgR =  p_YgX.T @ p_XgR
    return np.sum(p_YgR*p_R*np.log2(p_YgR/p_Y.reshape(-1,1)))

# compute I(R;Y) using distributions, version for use within IB algorithm (where p_YgR, p_R, and p_Y are already computed and passed in as arguments)
# def compute_I_RY_(p_YgR,p_R,p_Y):
#     return np.sum(p_YgR*p_R*np.log2(p_YgR/p_Y.reshape(-1,1)))

# compute I(X;R) using empirical list of choice probabilities for each trial (for model- or algorithm-computed choice probabilities only)
def compute_I_XR_emp(p_RgX_emp):
    N_trials = p_RgX_emp.shape[0]
    p_RX_emp = p_RgX_emp * (1/N_trials)
    return np.sum(p_RX_emp*np.log2(p_RgX_emp/np.sum(p_RX_emp, axis=0, keepdims=True)))

# compute I(R;Y) using empirical list of choice probabilities and empirical list of Y for each trial (for model- or algorithm-computed choice probabilities only)
def compute_I_RY_emp(p_RgX_emp, Yemp):
    N_trials = p_RgX_emp.shape[0]
    N_Y = p_RgX_emp.shape[1]
    Yemp_oh = np.eye(N_Y)[Yemp.astype(int)]
    p_Y_emp = np.sum(Yemp_oh,axis=0,keepdims=True) / N_trials
    p_RX_emp = p_RgX_emp * (1/N_trials)
    p_R_emp = np.sum(p_RX_emp, axis=0, keepdims=True)
    p_XgR_emp = p_RX_emp / p_R_emp
    p_YgR_emp = Yemp_oh.T @ p_XgR_emp
    return np.sum(p_YgR_emp * p_R_emp * np.log2(p_YgR_emp/p_Y_emp.reshape(-1,1)))

# compute IB cost function value (standalone version)
def compute_IB_fn(beta,p_RgX,p_XgY,p_Y):
        Ixr = compute_I_XR(p_RgX,p_XgY,p_Y)
        Iry = compute_I_RY(p_RgX,p_XgY,p_Y)
        return Ixr - beta*Iry

# compute IB cost function value (for use within the IB algorithm)
# def compute_IB_fn_(beta,p_RgX,p_X,p_R,p_YgR,p_Y):
#         Ixr = compute_I_XR_(p_RgX,p_X,p_R)
#         Iry = compute_I_RY_(p_YgR,p_R,p_Y)
#         return Ixr - beta*Iry

# from embo
def compute_upper_bound(IX, IY):
    """Remove all points in an IB sequence that would make it nonmonotonic.

    This is a post-processing step that is needed after computing an
    IB sequence (defined as a sequence of (IX, IY) pairs),
    to remove the random fluctuations in the result induced by the AB
    algorithm getting stuck in local minima.

    Parameters
    ----------
    IX : array
        I(M:X) values
    IY : array 
        I(M:Y) values

    Returns
    -------
    array (n x 2)
        (I(M:X), I(M:Y)) coordinates of the IB bound after ensuring monotonic progression (with increasing beta) in both coordinates.
    array ()
        The indices of the points selected to ensure monotonic progression

    """
    points = np.vstack((IX,IY)).T
    selected_idxs = [0]

    for idx in range(1,points.shape[0]):
        if points[idx,0]>points[selected_idxs[-1],0] and points[idx,1]>=points[selected_idxs[-1],1]:
            selected_idxs.append(idx)
            
    upper_bound = points[selected_idxs, :]

    return upper_bound, selected_idxs

# a function for attempting to sort the rows of p_YgR (for an IB solution) so that the maximum value in each row is on the diagonal
# this is for the purposes of comparing with the softmax solution
def get_map_inds(p_YgR,axis=1):
    p_YgR_copy = p_YgR.copy()
    if axis == 1:
        p_YgR_copy = p_YgR_copy.T
    mask = np.zeros_like(p_YgR_copy,dtype=bool)
    map_inds = []
    for ii, p_Ygr in enumerate(p_YgR_copy):
        p_Ygr_copy = p_Ygr.copy()
        p_Ygr_copy[map_inds] = -np.inf
        map_inds.append(np.argmax(p_Ygr_copy))
        mask[ii,np.argmax(p_Ygr_copy)] = True
    if axis == 1:
        mask = mask.T
    inv_map_inds = np.zeros_like(map_inds)
    for ii, ind in enumerate(map_inds):
        inv_map_inds[ind] = ii
    return map_inds, inv_map_inds, mask

# a function for solving the IB problem for a given beta, with multiple random initializations
def solve_IB(beta, p_XgY, p_Y, iterlimit=100000, init='random', N_inits=3, betastar=None, base_seed=209, ib_num=0, return_dict=True):

    rng = np.random.default_rng(base_seed + ib_num)

    N_y = p_XgY.shape[1]
    N_x = p_XgY.shape[0]

    if p_Y is None:
        p_Y = np.array([1/N_y]*N_y)

    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_YgX = p_XY / p_X

    IB_fxn_vals = []
    p_RgXs = []
    p_Rs = []
    p_YgRs = []
    for ii in range(N_inits):

        p_R = (np.ones(N_y) / N_y).reshape(1,-1)
        if init == 'random':
            p_RX = np.exp(rng.random((N_x,N_y)))
            p_RgX = p_RX / np.sum(p_RX,axis=1,keepdims=True)
        elif init == 'softmax':
            if betastar is None:
                betastar = beta
            p_RgX = np.exp(betastar*p_YgX) / np.sum(np.exp(betastar*p_YgX),axis=1,keepdims=True)
        p_XgR = (p_RgX * p_X) / p_R
        p_YgR =  p_YgX.T @ p_XgR
        IB_fxn = compute_IB_fn(beta,p_RgX,p_XgY,p_Y)
        delta = 1
        iters = 0
        while (delta > 1e-15) or (iters < 100):
            # store last copy
            IB_fxn_last = IB_fxn

            # update p(r|x)
            p_RX = p_R * np.exp(beta*(p_YgX @ np.log(p_YgR))) # we can simplify the original update rule (below) to this
            # p_RX = p_R * np.exp(-beta*np.sum(p_YgX[:,:,np.newaxis] * (np.log(p_YgX[:,:,np.newaxis]) - np.log(p_YgR[np.newaxis,:,:])),axis=1))
            p_RgX = p_RX / np.sum(p_RX,axis=1,keepdims=True)

            # update p(r)
            p_R = p_X.T @ p_RgX

            #update p(y|r)
            p_XgR = (p_RgX * p_X) / p_R
            p_YgR =  p_YgX.T @ p_XgR

            IB_fxn = compute_IB_fn(beta,p_RgX,p_XgY,p_Y)

            delta = np.abs(IB_fxn - IB_fxn_last)
            iters += 1
            if iters == iterlimit:
                break
        IB_fxn_vals.append(IB_fxn.copy())
        p_RgXs.append(p_RgX.copy())
        p_Rs.append(p_R.copy())
        p_YgRs.append(p_YgR.copy())

    min_ind = np.argmin(np.array(IB_fxn_vals))
    p_RgX = p_RgXs[min_ind]
    p_R = p_Rs[min_ind]
    p_YgR = p_YgRs[min_ind]
    # p_RY = p_YgR * p_R

    H_R = -np.sum(p_R * np.log2(p_R))
    I_XR = compute_I_XR(p_RgX,p_XgY,p_Y)
    I_RY = compute_I_RY(p_RgX,p_XgY,p_Y)

    if return_dict:
        out = {
            'I_XR': I_XR,
            'I_RY': I_RY,
            'beta': beta,
            'H_R': H_R,
            'p_RgX': p_RgX,
            'p_YgR': p_YgR,
            'p_R': p_R,
            'betastar': betastar
        }
        return out
    else:
        return [I_XR, I_RY, beta, H_R, p_RgX, p_YgR, p_R, betastar]

# a function for solving the IB problem for a given beta, with multiple random initializations
# this is function is intended to be used within a multiprocessing call, so it takes in all of the arguments as a single tuple
def solve_IB_mp(args):
    beta, p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib_num = args
    return solve_IB(beta, p_XgY, p_Y, iterlimit=iterlimit, init=init, N_inits=N_inits, betastar=betastar, base_seed=base_seed, ib_num=ib_num, return_dict=False)

# beta, p_XgY, p_Y, iterlimit, init, N_inits, betastar, base_seed, ib_num = args
def get_IB_bound(p_XgY,p_Y,N_b=1000,max_b=50,iterlimit=100000,beta_array=None,init='random', N_inits=3, betastar_array=None, N_threads=1, base_seed=209, enforce_monotonic=True, sort_by_beta=True):
    if beta_array is None:
        beta_array = np.linspace(max_b/N_b,max_b,N_b)
    # if betastar_array is None:
    #     betastar_array = np.zeros(N_b)

    with mp.Pool(processes=N_threads) as pool:
        results = [pool.apply_async(solve_IB_mp, args=((beta_array[ib], p_XgY, p_Y, iterlimit, init, N_inits, betastar_array[ib] if betastar_array is not None else None, base_seed, ib),)) for ib in range(N_b)]
        results = [res.get() for res in results]
    I_XR = [res[0] for res in results]
    I_RY = [res[1] for res in results]
    betas = [res[2] for res in results]
    H_Rs = [res[3] for res in results]
    p_RgXs = [res[4] for res in results]
    p_YgRs = [res[5] for res in results]
    p_Rs = [res[6] for res in results]

    # to ensure that the points on the IB curve are in order of increasing beta, we can sort the results by beta before returning
    # this can result in non-monotonicity of the IB bound when doing a point-by-point comparison with the softmax solution 
    if sort_by_beta:
        sort_indices = np.argsort(betas)
    else:
        sort_indices = np.arange(len(betas))
    out = {
        'I_XR': np.array(I_XR)[sort_indices],
        'I_RY': np.array(I_RY)[sort_indices],
        'beta': np.array(betas)[sort_indices],
        'H_R': np.array(H_Rs)[sort_indices],
        'p_RgX': np.array(p_RgXs)[sort_indices],
        'p_YgR': np.array(p_YgRs)[sort_indices],
        'p_R': np.array(p_Rs)[sort_indices]
    }
    if enforce_monotonic:
        ub, idxs = compute_upper_bound(out['I_XR'], out['I_RY'])
        out['I_XR'] = np.squeeze(ub[:,0])
        out['I_RY'] = np.squeeze(ub[:,1])
        out['beta'] = out['beta'][idxs]
        out['H_R'] = out['H_R'][idxs]
        out['p_RgX'] = out['p_RgX'][idxs]
        out['p_YgR'] = out['p_YgR'][idxs]
        out['p_R'] = out['p_R'][idxs]
    # return np.array(I_XR)[sort_indices], np.array(I_RY)[sort_indices], np.array(betas)[sort_indices], np.array(H_Rs)[sort_indices]
    return out

# get the IB measures for the IB optimal choice policy (computed using the true p(X,Y)) for a particular trial set (empirical p(X,Y))
def get_IB_emp(beta,Xemp,Xset,Yemp,p_XgY_true,p_Y_true, iterlimit=100000, init='random', N_inits=3, betastar=None, base_seed=209, ib_num=0,return_dict=True):

    N_trials = Xemp.shape[0]
    N_Y = p_Y_true.shape[1]
    Yemp_oh = np.eye(N_Y)[Yemp.astype(int)]
    p_Y_emp = np.sum(Yemp_oh,axis=0,keepdims=True) / N_trials

    Xinds = np.zeros(Xemp.shape[0], dtype=int)
    for ii, x in enumerate(Xset):
        matches = np.all(Xemp == x, axis=1)
        Xinds[matches] = ii

    out = solve_IB(beta, p_XgY_true, p_Y_true, iterlimit=iterlimit, init=init, N_inits=N_inits, betastar=betastar, base_seed=base_seed, ib_num=ib_num, return_dict=False)
    p_RgX = out[4]
    p_RgX_emp = p_RgX[Xinds,:]
    p_RX_emp = p_RgX_emp * (1/N_trials)
    p_R_emp = np.sum(p_RX_emp, axis=0, keepdims=True)
    p_XgR_emp = p_RX_emp / p_R_emp
    p_YgR_emp = Yemp_oh.T @ p_XgR_emp
    I_XR_emp = compute_I_XR_emp(p_RgX_emp)
    I_RY_emp = compute_I_RY_emp(p_RgX_emp, Yemp)
    H_R_emp = -np.sum(p_R_emp * np.log2(p_R_emp))
    if return_dict:
        out = {
            'I_XR': I_XR_emp,
            'I_RY': I_RY_emp,
            'beta': beta,
            'H_R': H_R_emp,
            'p_RgX': p_RgX_emp,
            'p_YgR': p_YgR_emp,
            'p_R': p_R_emp,
            'betastar': betastar
        }
        return out
    else:
        return [I_XR_emp, I_RY_emp, beta, H_R_emp, p_RgX_emp, p_YgR_emp, p_R_emp, betastar]

# same as above, but for multiprocessing
def get_IB_emp_mp(args):
    return get_IB_emp(*args)

def get_IB_bound_emp(Xemp,Xset,Yemp,p_XgY_true,p_Y_true,N_b=1000,max_b=50,beta_array=None,iterlimit=100000,init='random', N_inits=3, betastar_array=None, N_threads=1, base_seed=209, enforce_monotonic=True, sort_by_beta=True):
    if beta_array is None:
        beta_array = np.linspace(max_b/N_b,max_b,N_b)
    # if betastar_array is None:
    #     betastar_array = np.zeros(N_b)

    with mp.Pool(processes=N_threads) as pool:
        results = [pool.apply_async(get_IB_emp_mp, args=((beta_array[ib], Xemp, Xset, Yemp, p_XgY_true, p_Y_true, iterlimit, init, N_inits, betastar_array[ib] if betastar_array is not None else None, base_seed, ib, False),)) for ib in range(N_b)]
        results = [res.get() for res in results]
    I_XR_emp = [res[0] for res in results]
    I_RY_emp = [res[1] for res in results]
    betas = [res[2] for res in results]
    H_Rs = [res[3] for res in results]
    p_RgXs = [res[4] for res in results]
    p_YgRs = [res[5] for res in results]
    p_Rs = [res[6] for res in results]

    # to ensure that the points on the IB curve are in order of increasing beta, we can sort the results by beta before returning
    if sort_by_beta:
        sort_indices = np.argsort(betas)
    else:
        sort_indices = np.arange(len(betas))
    out = {
        'I_XR': np.array(I_XR_emp)[sort_indices],
        'I_RY': np.array(I_RY_emp)[sort_indices],
        'beta': np.array(betas)[sort_indices],
        'H_R': np.array(H_Rs)[sort_indices],
        'p_RgX': np.array(p_RgXs)[sort_indices],
        'p_YgR': np.array(p_YgRs)[sort_indices],
        'p_R': np.array(p_Rs)[sort_indices],
    }
    if enforce_monotonic:
        ub, idxs = compute_upper_bound(out['I_XR'], out['I_RY'])
        out['I_XR'] = np.squeeze(ub[:,0])
        out['I_RY'] = np.squeeze(ub[:,1])
        out['beta'] = out['beta'][idxs]
        out['H_R'] = out['H_R'][idxs]
        out['p_RgX'] = out['p_RgX'][idxs]
        out['p_YgR'] = out['p_YgR'][idxs]
        out['p_R'] = out['p_R'][idxs]
    # return np.array(I_XR_emp)[sort_indices], np.array(I_RY_emp)[sort_indices], np.array(betas)[sort_indices], np.array(H_Rs)[sort_indices]
    return out

# get IB measures for the softmax policy on the true p(X,Y)
def get_softmax_IB(betastar,p_XgY,p_Y,return_dict=True):
    N_Y = p_XgY.shape[1]
    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    p_YgX = p_XY / p_X

    p_RgX = np.exp(betastar*p_YgX) / np.sum(np.exp(betastar*p_YgX),axis=1,keepdims=True)
    p_XR = p_RgX*p_X
    p_R = np.sum(p_XR, axis=0, keepdims=True)
    p_XgR = (p_RgX * p_X) / p_R
    p_YgR =  p_YgX.T @ p_XgR

    accuracy = np.trace(p_YgR * p_R)
    alphastar = np.log(accuracy/((1-accuracy)/(N_Y-1))) # alpha* and this expression itself is only valid when p(Y) is uniform
    beta = betastar / alphastar
    I_XR = compute_I_XR(p_RgX,p_XgY,p_Y)
    I_RY = compute_I_RY(p_RgX,p_XgY,p_Y)
    H_R = -np.sum(p_R * np.log2(p_R))
    if return_dict:
        out = {
            'I_XR': I_XR,
            'I_RY': I_RY,
            'betastar': betastar,
            'H_R': H_R,
            'accuracy': accuracy,
            'alphastar': alphastar,
            'beta': beta,
            'p_RgX': p_RgX,
            'p_YgR': p_YgR,
            'p_R': p_R
        }
        return out
    else:
        return I_XR, I_RY, betastar, H_R, accuracy, alphastar, beta, p_RgX, p_YgR, p_R

def get_softmax_IB_curve(p_XgY,p_Y,N_b=1000,max_b=50,betastar_array=None):
    if betastar_array is None:
        betastar_array = np.linspace(max_b/N_b,max_b,N_b)
    I_XRs, I_RYs, H_Rs, accuracies, alphastars, betas, p_RgXs, p_YgRs, p_Rs = [], [], [], [], [], [], [], [], []
    for betastar in betastar_array:
        out = get_softmax_IB(betastar,p_XgY,p_Y,return_dict=False)
        I_XRs.append(out[0])
        I_RYs.append(out[1])
        H_Rs.append(out[3])
        accuracies.append(out[4])
        alphastars.append(out[5])
        betas.append(out[6]),
        p_RgXs.append(out[7])
        p_YgRs.append(out[8])
        p_Rs.append(out[9])
    out = {
        'I_XR': np.array(I_XRs),
        'I_RY': np.array(I_RYs),
        'betastar': betastar_array,
        'H_R': np.array(H_Rs),
        'accuracy': np.array(accuracies),
        'alphastar': np.array(alphastars),
        'beta': np.array(betas),
        'p_RgX': np.array(p_RgXs),
        'p_YgR': np.array(p_YgRs),
        'p_R': np.array(p_Rs)
    }
    return out
    # return np.array(I_XRs), np.array(I_RYs), betastar_array, np.array(H_Rs), np.array(accuracies), np.array(alphastars), np.array(betas)

def get_softmax_IB_emp(betastar,Xemp,Yemp,p_YgX_emp,return_dict=True):
    N_trials = Xemp.shape[0]
    N_Y = p_YgX_emp.shape[1]
    Yemp_oh = np.eye(N_Y)[Yemp.astype(int)]
    p_RgX_emp = np.exp(betastar*p_YgX_emp) / np.sum(np.exp(betastar*p_YgX_emp),axis=1,keepdims=True)
    p_RX_emp = p_RgX_emp * (1/N_trials)
    p_R_emp = np.sum(p_RX_emp, axis=0, keepdims=True)
    p_XgR_emp = p_RX_emp / p_R_emp
    p_YgR_emp = Yemp_oh.T @ p_XgR_emp
    I_XR_emp = compute_I_XR_emp(p_RgX_emp)
    I_RY_emp = compute_I_RY_emp(p_RgX_emp, Yemp)
    H_R_emp = -np.sum(p_R_emp * np.log2(p_R_emp))
    accuracy = np.trace(p_YgR_emp * p_R_emp)
    alphastar = np.log(accuracy/((1-accuracy)/(N_Y-1))) # alpha* and this expression itself is only valid when p(Y) is uniform
    beta = betastar / alphastar
    if return_dict:
        out = {
            'I_XR': I_XR_emp,
            'I_RY': I_RY_emp,
            'betastar': betastar,
            'H_R': H_R_emp,
            'accuracy': accuracy,
            'alphastar': alphastar,
            'beta': beta,
            'p_RgX': p_RgX_emp,
            'p_YgR': p_YgR_emp,
            'p_R': p_R_emp
        }
        return out
    else:
        return I_XR_emp, I_RY_emp, betastar, H_R_emp, accuracy, alphastar, beta, p_RgX_emp, p_YgR_emp, p_R_emp

def get_softmax_IB_curve_emp(Xemp,Yemp,p_YgX_emp,N_b=1000,max_b=50,betastar_array=None):
    if betastar_array is None:
        betastar_array = np.linspace(max_b/N_b,max_b,N_b)
    I_XRs, I_RYs, H_Rs, accuracies, alphastars, betas, p_RgXs, p_YgRs, p_Rs = [], [], [], [], [], [], [], [], []
    for betastar in betastar_array:
        out = get_softmax_IB_emp(betastar,Xemp,Yemp,p_YgX_emp,return_dict=False)
        I_XRs.append(out[0])
        I_RYs.append(out[1])
        H_Rs.append(out[3])
        accuracies.append(out[4])
        alphastars.append(out[5])
        betas.append(out[6])
        p_RgXs.append(out[7])
        p_YgRs.append(out[8])
        p_Rs.append(out[9])
    out = {
        'I_XR': np.array(I_XRs),
        'I_RY': np.array(I_RYs),
        'betastar': betastar_array,
        'H_R': np.array(H_Rs),
        'accuracy': np.array(accuracies),
        'alphastar': np.array(alphastars),
        'beta': np.array(betas),
        'p_RgX': np.array(p_RgXs),
        'p_YgR': np.array(p_YgRs),
        'p_R': np.array(p_Rs)
    }
    # return np.array(I_XRs), np.array(I_RYs), betastar_array, np.array(H_Rs), np.array(accuracies), np.array(alphastars), np.array(betas)
    return out

def get_general_IB(p_RgX,p_XgY,p_Y,return_dict=True):
    I_XR = compute_I_XR(p_RgX,p_XgY,p_Y)
    I_RY = compute_I_RY(p_RgX,p_XgY,p_Y)
    p_RY = (p_RgX.T @ p_XgY) * p_Y
    p_R = np.sum(p_RY, axis=1, keepdims=True)
    H_R = -np.sum(p_R * np.log2(p_R))
    accuracy = np.trace(p_RY)
    p_YgR = p_RY / p_R
    if return_dict:
        out = {
            'I_XR': I_XR,
            'I_RY': I_RY,
            'H_R': H_R,
            'accuracy': accuracy,
            'p_RgX': p_RgX,
            'p_YgR': p_YgR,
            'p_R': p_R
        }
        return out
    else:
        return I_XR, I_RY, H_R, accuracy, p_RgX, p_YgR, p_R

def get_general_IB_curve(param_vals,p_XgY,p_Y,P_RgX_fn):
    #P_RgX_fn is a function that only takes in a parameter value, so any other arguments must be included in the function defintion of P_RgX_fn (e.g. using a lambda function to "pre-load" the other arguments)
    I_XRs, I_RYs, H_Rs, accuracies, p_RgXs, p_YgRs, p_Rs = [], [], [], [], [], [], []
    for param_val in param_vals:
        p_RgX = P_RgX_fn(param_val)
        out = get_general_IB(p_RgX,p_XgY,p_Y,return_dict=False)
        I_XRs.append(out[0])
        I_RYs.append(out[1])
        H_Rs.append(out[2])
        accuracies.append(out[3])
        p_RgXs.append(p_RgX)
        p_YgRs.append(out[5])
        p_Rs.append(out[6])
    out = {
        'I_XR': np.array(I_XRs),
        'I_RY': np.array(I_RYs),
        'H_R': np.array(H_Rs),
        'accuracy': np.array(accuracies),
        'p_RgX': np.array(p_RgXs),
        'p_YgR': np.array(p_YgRs),
        'p_R': np.array(p_Rs)
    }
    # return np.array(I_XRs), np.array(I_RYs), np.array(H_Rs), np.array(accuracies)
    return out

def get_general_IB_emp(Xemp,Yemp,p_RgX_emp,return_dict=True):
    N_trials = Xemp.shape[0]
    N_Y = p_RgX_emp.shape[1]
    Yemp_oh = np.eye(N_Y)[Yemp.astype(int)]
    p_RX_emp = p_RgX_emp * (1/N_trials)
    p_R_emp = np.sum(p_RX_emp, axis=0, keepdims=True)
    p_XgR_emp = p_RX_emp / p_R_emp
    p_YgR_emp = Yemp_oh.T @ p_XgR_emp
    I_XR_emp = compute_I_XR_emp(p_RgX_emp)
    I_RY_emp = compute_I_RY_emp(p_RgX_emp, Yemp)
    H_R_emp = -np.sum(p_R_emp * np.log2(p_R_emp))
    accuracy = np.trace(p_YgR_emp * p_R_emp)
    if return_dict:
        out = {
            'I_XR': I_XR_emp,
            'I_RY': I_RY_emp,
            'H_R': H_R_emp,
            'accuracy': accuracy,
            'p_RgX': p_RgX_emp,
            'p_YgR': p_YgR_emp,
            'p_R': p_R_emp
        }
        return out
    else:
        return I_XR_emp, I_RY_emp, H_R_emp, accuracy, p_RgX_emp, p_YgR_emp, p_R_emp

def get_general_IB_curve_emp(param_vals,Xemp,Yemp,p_RgX_emp_fn):
    I_XRs, I_RYs, H_Rs, accuracies, p_RgXs, p_YgRs, p_Rs = [], [], [], [], [], [], []
    for param_val in param_vals:
        p_RgX_emp = p_RgX_emp_fn(param_val)
        out = get_general_IB_emp(Xemp,Yemp,p_RgX_emp, return_dict=False)
        I_XRs.append(out[0])
        I_RYs.append(out[1])
        H_Rs.append(out[2])
        accuracies.append(out[3])
        p_RgXs.append(p_RgX_emp)
        p_YgRs.append(out[5])
        p_Rs.append(out[6])
    out = {
        'I_XR': np.array(I_XRs),
        'I_RY': np.array(I_RYs),
        'H_R': np.array(H_Rs),
        'accuracy': np.array(accuracies),
        'p_RgX': np.array(p_RgXs),
        'p_YgR': np.array(p_YgRs),
        'p_R': np.array(p_Rs)
    }
    return out

### Functions for computing likelihoods and posteriors for the horse prediction task ###

def get_beads_Xset(wsize):
    # get all possible bead sequences for a given window size
    return np.array([seq for seq in product([0,1],repeat=wsize)])

# function for generating bead sequence (2 jar types, 2 bead types)
def gen_bead_seq(n,h,P):
    # n is the length of the bead sequence
    # h is the hazard rate (probability of switching states on each trial)
    # P is the 2x1 array of probabilities of observing the predominant bead type given the current state
    states = np.zeros((n))
    obs = np.zeros((n))
        
    state_cur = np.random.choice([0,1])
    for i in range(n):
        states[i] = state_cur
        if np.random.uniform() < P[state_cur]:
            obs[i] = state_cur
        else:
            obs[i] = 1-state_cur
                
        if np.random.uniform() < h:
            state_cur = 1-state_cur
                
    return states.astype(int), obs.astype(int)

def P_beads_g_jar(bead_seqs,E,H):
    bead_seqs_oh = np.transpose(np.eye(E.shape[1])[bead_seqs],(0,2,1))
    L = np.matmul(E,bead_seqs_oh)
    H = H.T
    P = np.ones((bead_seqs.shape[0],E.shape[1],1))
    for ib in range(1,bead_seqs.shape[1]):
        P = np.matmul(H,np.multiply(L[:,:,ib-1][:,:,np.newaxis],P))
    P = np.multiply(L[:,:,bead_seqs.shape[1]-1][:,:,np.newaxis],P)
    return np.squeeze(P)

def P_jar_g_beads(bead_seqs,E,H,p_jar):
    p_XgY = P_beads_g_jar(bead_seqs,E,H)
    # N_y = p_XgY.shape[1]
    p_XY = p_XgY * p_jar
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    return p_XY / p_X

# compute likelihood distributions for the horse prediction task
def llr2probs_4shapes(llr,p1):
    
    # INPUT: array of llr for each of the 4 shapes, plus a parameter p1
    # OUTPUT: pdist1 is P(shapes|state 1), pdist2 is P(shapes|state 2)
    
    # p1 is the likelihood of shape 1 given state 1 (blue bar on the far left)
    pdist1 = np.zeros((4))
    pdist2 = np.zeros((4))

    lr = 10**llr

    pdist1[0] = p1
    pdist2[0] = lr[3]*p1
    pdist1[3] = pdist2[0]
    pdist2[3] = p1

    pdist1[1] = (1-(lr[3]+1)*p1) / (lr[2]+1)
    pdist2[1] = lr[2]*pdist1[1]
    pdist1[2] = pdist2[1]
    pdist2[2] = pdist1[1]

    return pdist1, pdist2

# convert a an array of shape combination codes to array of 4 integers, where each integer is the count of a particular shape in the combination
# def split_to_four_digits(arr):
#     # order of input array is [shape4, shape3, shape2, shape1]
#     # need to convert to [shape1, shape2, shape3, shape4] eventually
#     arr = arr.astype(int)
#     result = np.zeros((arr.shape[0], 4), dtype=int)
#     for i, num in enumerate(arr):
#         digits = list(str(num).zfill(4))
#         result[i] = [int(d) for d in digits]
#     result = result[:, ::-1]  # reverse the order to get [shape1, shape2, shape3, shape4]
#     return result

def split_to_digits(arr):
    # split an array of integers into an array of digits, where each digit is a shape type
    arr = arr.astype(int)
    N_shapes = len(str(np.max(arr)))
    result = np.zeros((arr.shape[0], N_shapes), dtype=int)
    for i, num in enumerate(arr):
        digits = list(str(num).zfill(N_shapes))
        result[i] = [int(d) for d in digits]
    result = result[:, ::-1]  # reverse the order
    return result

def get_shapes_Xset(encoding='full'):
    # get all possible shape combinations (with replacement) for 5 shapes and 4 shape types
    if encoding == 'full':
        shape_combs = list(combinations_with_replacement(range(4), 5))
        Xset = np.array(shape_combs)
        # change to encode how many of each shape type are present in order of num_shape1, num_shape2, num_shape3, num_shape4
        Xset_counts = np.zeros((Xset.shape[0], 4), dtype=int)
        for i, comb in enumerate(Xset):
            for shape in comb:
                Xset_counts[i, shape] += 1
    if encoding == 'equal-weights':
        Xset_counts = np.arange(5+1)
        Xset_counts = np.array([Xset_counts, 5-Xset_counts]).T
    if encoding == 'ignore-weak':
        shape_combs = list(combinations_with_replacement(range(4), 5))
        Xset = np.array(shape_combs)
        # change to encode how many of each shape type are present in order of num_shape1, num_shape2, num_shape3, num_shape4
        Xset_counts = np.zeros((Xset.shape[0], 4), dtype=int)
        for i, comb in enumerate(Xset):
            for shape in comb:
                Xset_counts[i, shape] += 1
        # if shape 1 or shape 4 is present, set count of shapes 2 and 3 to 0
        for i in range(Xset_counts.shape[0]):
            if Xset_counts[i,0] > 0 or Xset_counts[i,3] > 0:
                Xset_counts[i,1] = 0
                Xset_counts[i,2] = 0
        # only get unique shape combs after ignoring weak shapes
        Xset_counts = np.unique(Xset_counts, axis=0)
    return Xset_counts

def gen_horses_trial_set(weakLLR,WSratio,p1,num_trials,num_shapes):
    llr = np.array([-weakLLR*WSratio, -weakLLR, weakLLR, weakLLR*WSratio])
    pdist1, pdist2 = llr2probs_4shapes(llr, p1)
    rng = np.random.default_rng()
    shape_inds = range(len(llr))
    trial_arrays = []
    trial_codes = []
    
    # state 1, so half of the number of trials
    for ii in range(int(num_trials/2)):
        trial_shapes = rng.choice(shape_inds,num_shapes,p=pdist1,shuffle=False)
        trial_arrays.append(trial_shapes)
        trial_codes.append((10**trial_shapes).sum())

    # state 2
    for ii in range(int(num_trials/2)):
        trial_shapes = rng.choice(shape_inds,num_shapes,p=pdist2,shuffle=False)
        trial_arrays.append(trial_shapes)
        trial_codes.append((10**trial_shapes).sum())

    trial_shapes = split_to_digits(np.array(trial_codes))
    return trial_shapes, np.array(trial_codes)

# compute likelihood of shape combination given horse
def P_shapecomb_g_horse(X, base, multiple, p1):
    optimal_llr = np.array([-base*multiple, -base, base, base*multiple]) 
    pdist1, pdist2 = llr2probs_4shapes(optimal_llr, p1)
    unique_perms = factorial(5) / np.array([factorial(X[ii]).prod() for ii in range(X.shape[0])])
    return np.column_stack(((pdist1**X).prod(axis=1)*unique_perms,(pdist2**X).prod(axis=1)*unique_perms))

# compute likelihood of shape combination given horse, with equal weighting of shapes 1&2 and shapes 3&4
def P_shapecomb_g_horse_ew(X, base, multiple, p1):
    optimal_llr = np.array([-base*multiple, -base, base, base*multiple]) 
    pdist1, pdist2 = llr2probs_4shapes(optimal_llr, p1)
    # pdist1_ew = np.array([pdist1[0]+pdist1[1], pdist1[0]+pdist1[1], pdist1[2]+pdist1[3],  pdist1[2]+pdist1[3]])
    # pdist2_ew = np.array([pdist2[0]+pdist2[1], pdist2[0]+pdist2[1], pdist2[2]+pdist2[3],  pdist2[2]+pdist2[3]])
    pdist1_ew = np.array([pdist1[0]+pdist1[1], pdist1[2]+pdist1[3]])
    pdist2_ew = np.array([pdist2[0]+pdist2[1], pdist2[2]+pdist2[3]])
    # unique_perms = factorial(5) / np.array([factorial(X[ii][0]+X[ii][1]) * factorial(X[ii][2]+X[ii][3]) for ii in range(X.shape[0])])
    unique_perms = factorial(5) / np.array([factorial(X[ii]).prod() for ii in range(X.shape[0])])
    return np.column_stack(((pdist1_ew**X).prod(axis=1)*unique_perms,(pdist2_ew**X).prod(axis=1)*unique_perms))

# compute likelihood of shape combination given horse, with ignoring the weak shapes (shapes 2 and 3) when strong shapes (shapes 1 and 4) are present
def P_shapecomb_g_horse_iw(X, base, multiple, p1):
    optimal_llr = np.array([-base*multiple, -base, base, base*multiple]) 
    pdist1, pdist2 = llr2probs_4shapes(optimal_llr, p1)
    weak_prob_strong = pdist1[1]+pdist1[2]
    likelihood1 = np.array([(pdist1**X[ii]).prod() if (X[ii,0]==0 and X[ii,3]==0) else (pdist1**X[ii]).prod() * weak_prob_strong**(5-X[ii,0]-X[ii,3]) for ii in range(X.shape[0])])
    likelihood2 = np.array([(pdist2**X[ii]).prod() if (X[ii,0]==0 and X[ii,3]==0) else (pdist2**X[ii]).prod() * weak_prob_strong**(5-X[ii,0]-X[ii,3]) for ii in range(X.shape[0])])
    unique_perms = factorial(5) / np.array([factorial(X[ii]).prod() if (X[ii,0]==0 and X[ii,3]==0) else factorial(np.array([X[ii,0], X[ii,3], 5-X[ii,0]-X[ii,3]])).prod() for ii in range(X.shape[0])])
    return np.column_stack((likelihood1*unique_perms, likelihood2*unique_perms))
    # Xstrong = X.copy()
    # Xstrong[:,1] = 0
    # Xstrong[:,2] = 0
    # Xiw = np.array([X[ii] if (X[ii,0]==0 and X[ii,3]==0) else Xstrong[ii] for ii in range(X.shape[0])])
    # unique_perms = np.array([factorial(Xiw[ii].sum()) for ii in range(Xiw.shape[0])]) / np.array([factorial(Xiw[ii]).prod() for ii in range(Xiw.shape[0])])
    # return np.column_stack(((pdist1**Xiw).prod(axis=1)*unique_perms,(pdist2**Xiw).prod(axis=1)*unique_perms))

# compute posterior probability of horse given shape combination
def P_horse_g_shapecomb(X, base, multiple, p1, p_Y, P_XgY=None):
    if P_XgY is None:
        p_XgY = P_shapecomb_g_horse(X, base, multiple, p1)
    else:
        p_XgY = P_XgY(X, base, multiple, p1)
    N_y = p_XgY.shape[1]

    p_XY = p_XgY * p_Y
    p_X = np.sum(p_XY, axis=1, keepdims=True)
    return p_XY / p_X

### Functions for pyschometric analyses

def get_sj_psychometric_plot(sj_pmf_df,strat_post_prob_df,ax):
    Nmax = sj_pmf_df['Ntrials'].max()
    for ii, Xun in enumerate(sj_pmf_df['X'].unique()):
       postconds = (strat_post_prob_df['X'] == Xun)
       sjconds = (sj_pmf_df['X'] == Xun)
       ax.plot(
            strat_post_prob_df.loc[postconds, 'posterior_prob'].iat[0],
            sj_pmf_df.loc[sjconds, 'choice_prob'].iat[0],
            'ok', markersize=3.5, markeredgewidth=0, markeredgecolor='none', alpha=sj_pmf_df.loc[sjconds, 'Ntrials'].iat[0]/Nmax
        )
       
def get_IB_predicted_psychometric_plot(Ixr,bound,color,ax):
    betastar = np.interp(Ixr, bound['I_XR'], bound['betastar'])
    p = np.arange(0,1.01,0.01).reshape(-1,1)
    sigmoid = softmax(betastar*np.concatenate((p,1-p),axis=1),axis=1)[:,0]
    ax.plot(p.squeeze(), sigmoid.squeeze(), color=color, linewidth=1, linestyle='dashed')

def get_psychometric_DKL(sj_pmf_df,strat_post_prob_df,sj_Ixr,bound):
    betastar = np.interp(sj_Ixr, bound['I_XR'], bound['betastar'])
    Ntrials = sj_pmf_df['Ntrials'].sum()
    DKLs = []
    for ii, Xun in enumerate(sj_pmf_df['X'].unique()):
        postcond = (strat_post_prob_df['X'] == Xun)
        sjcond = (sj_pmf_df['X'] == Xun)
        p_choice = sj_pmf_df.loc[sjcond, 'choice_prob'].iat[0]
        p_choice = np.array([p_choice, 1-p_choice])
        post_prob = strat_post_prob_df.loc[postcond, 'posterior_prob'].iat[0]
        post_prob = np.array([post_prob, 1-post_prob])
        p_pred = softmax(betastar * post_prob)
        p_x = sj_pmf_df.loc[sjcond, 'Ntrials'].iat[0] / Ntrials
        DKL = p_x * np.nansum(p_choice * np.log(p_choice / p_pred))
        DKLs.append(DKL)
    return np.nansum(DKLs)

### Functions for model fitting

def compute_neg_sum_log_likelihood(sj_choices,model_fn,params):
    N_choices = len(np.unique(sj_choices))
    choices_oh = np.eye(N_choices)[sj_choices.astype(int)]
    model_probs = model_fn(params)
    return -np.nansum(choices_oh*np.log(model_probs))

def compute_BIC(num_params,num_trials,negsumLL):
    return num_params*np.log(num_trials) + 2*negsumLL

# computes the empirical p(R|X) for a given subject, then lists the relevant p(R|X) by trial in the trial set
# X is the trial set, R is the subject's choices for each trial
def get_empirical_pRgX(X,R):
    NX = X.shape[0]
    Roh = np.eye(int(R.max()+1))[R.astype(int)]
    Xunique = np.unique(X)
    pRgX = np.zeros((X.shape[0],Roh.shape[1]))
    for ii, Xun in enumerate(Xunique):
        Xinds_ = np.where(X == Xun)[0]
        choices_oh = Roh[Xinds_]
        pRgX[Xinds_,:] = choices_oh.mean(axis=0)
    return pRgX

# computed in the same way as the negative sum of the LLs (trial-wise)
def compute_rel_entropy(sj_choice_probs,model_choice_probs):
    return np.nansum(sj_choice_probs*np.log(sj_choice_probs/model_choice_probs)) / sj_choice_probs.shape[0]

### Function for information efficiency versus parameter value analyses

def get_efficiency_vs_parameter_curve(target_param_vals,Ixr_param_vals,P_RgX_fn,Xemp,Yemp,Ixr_vals,opt_bound):
    efficiencies = []
    for param_val in target_param_vals:
        P_RgX_fn_ = lambda Ixr_param_val: P_RgX_fn(param_val, Ixr_param_val)
        ibcurve = get_general_IB_curve_emp(Ixr_param_vals,Xemp,Yemp,P_RgX_fn_)
        Iry_curve = np.interp(Ixr_vals, ibcurve['I_XR'], ibcurve['I_RY'])
        Iry_opt = np.interp(Ixr_vals,opt_bound['I_XR'], opt_bound['I_RY'])
        efficiencies.append(np.mean(Iry_curve / Iry_opt))
    return efficiencies