# influence functions for shapley values

def shapley_influence_function(Z, z_counts, W, v, psi, G, c_n, ics, measure):
    """
    Compute influence function for the given predictiveness measure

    @param Z the subsets of the power set with estimates
    @param W the matrix of weights
    @param v the estimated predictivness
    @param psi the estimated Shapley values
    @param G the constrained ls matrix
    @param c_n the constraints
    @param ics a list of all ics
    @param measure the predictiveness measure
    """
    import numpy as np

    ## compute contribution from estimating V
    Z_W = Z.transpose().dot(W)
    A_m = Z_W.dot(Z)
    A_m_inv = np.linalg.inv(A_m)
    phi_01 = A_m_inv.dot(Z_W).dot(ics)

    ## compute contribution from estimating Q
    qr_decomp = np.linalg.qr(G.transpose(), mode = 'complete')
    U_2 = qr_decomp[0][:, 3:(Z.shape[1])]
    V = U_2.transpose().dot(Z.transpose().dot(W).dot(Z)).dot(U_2)
    phi_02_shared_mat = (-1) * U_2.dot(np.linalg.inv(V))
    phi_02_uniq_vectors = np.array([(Z[z, :].dot(psi) - v[z]) * (U_2.transpose().dot(Z[z, :])) for z in range(Z.shape[0])]).transpose()
    phi_02_uniq = phi_02_shared_mat.dot(phi_02_uniq_vectors)
    phi_02 = np.repeat(phi_02_uniq, z_counts, axis=1)

    return {'contrib_v': phi_01, 'contrib_s': phi_02}


def shapley_se(shapley_ics, idx, gamma, na_rm = True):
    """
    Standard error for the desired Shapley value

    @param shapley_ics: all influence function estimates
    @param idx: the index of interest
    @param gamma: the constant for sampling
    @param na_rm: remove NaNs?

    @return the standard error corresponding to the shapley value at idx
    """
    import numpy as np
    if na_rm:
        var_v = np.nanvar(shapley_ics['contrib_v'][idx, :])
        var_s = np.nanvar(shapley_ics['contrib_s'][idx, :])
    else:
        var_v = np.var(shapley_ics['contrib_v'][idx, :])
        var_s = np.var(shapley_ics['contrib_s'][idx, :])
    se = np.sqrt(var_v / shapley_ics['contrib_v'].shape[1] + var_s / shapley_ics['contrib_s'].shape[1] / gamma)
    return se
