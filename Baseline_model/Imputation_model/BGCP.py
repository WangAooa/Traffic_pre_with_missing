import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from numpy.linalg import inv as inv

def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)
def cp_combine(var):
    return np.einsum('is, js, ts -> ijt', var[0], var[1], var[2])
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
def Compute_RMSE(var, var_hat):
    return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
def Compute_MAPE(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]
def Compute_MAE(var, var_hat):
    return np.sum(np.abs(var - var_hat)) / var.shape[0]

def cov_mat(mat):
    dim1, dim2 = mat.shape
    new_mat = np.zeros((dim2, dim2))
    mat_bar = np.mean(mat, axis = 0)
    for i in range(dim1):
        new_mat += np.einsum('i, j -> ij', mat[i, :] - mat_bar, mat[i, :] - mat_bar)
    return new_mat


def BGCP(dense_tensor, sparse_tensor, factor, burnin_iter, gibbs_iter):
    """Bayesian Gaussian CP (BGCP) decomposition."""

    dim = np.array(sparse_tensor.shape)
    rank = factor[0].shape[1]
    pos_train = np.where(sparse_tensor != 0)
    pos_test = np.where((dense_tensor != 0) & (sparse_tensor == 0))
    binary_tensor = np.zeros(dim)
    binary_tensor[pos_train] = 1

    beta0 = 1
    nu0 = rank
    mu0 = np.zeros((rank))
    W0 = np.eye(rank)
    tau = 1
    alpha = 1e-6
    beta = 1e-6

    factor_plus = []
    for k in range(len(dim)):
        factor_plus.append(np.zeros((dim[k], rank)))
    tensor_hat_plus = np.zeros(dim)
    for it in range(burnin_iter + gibbs_iter):
        for k in range(len(dim)):
            mat_bar = np.mean(factor[k], axis=0)
            var_mu_hyper = (dim[k] * mat_bar + beta0 * mu0) / (dim[k] + beta0)
            var_W_hyper = inv(inv(W0) + cov_mat(factor[k]) + dim[k] * beta0 / (dim[k] + beta0)
                              * np.outer(mat_bar - mu0, mat_bar - mu0))
            var_Lambda_hyper = wishart(df=dim[k] + nu0, scale=var_W_hyper, seed=None).rvs()
            var_mu_hyper = mvnrnd(var_mu_hyper, inv((dim[k] + beta0) * var_Lambda_hyper))

            if k == 0:
                var1 = kr_prod(factor[k + 2], factor[k + 1]).T
            elif k == 1:
                var1 = kr_prod(factor[k + 1], factor[k - 1]).T
            else:
                var1 = kr_prod(factor[k - 1], factor[k - 2]).T
            var2 = kr_prod(var1, var1)
            var3 = (tau * np.matmul(var2, ten2mat(binary_tensor, k).T).reshape([rank, rank, dim[k]])
                    + np.dstack([var_Lambda_hyper] * dim[k]))
            var4 = (tau * np.matmul(var1, ten2mat(sparse_tensor, k).T)
                    + np.dstack([np.matmul(var_Lambda_hyper, var_mu_hyper)] * dim[k])[0, :, :])
            for i in range(dim[k]):
                var_Lambda = var3[:, :, i]
                inv_var_Lambda = inv((var_Lambda + var_Lambda.T) / 2)
                factor[k][i, :] = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
        tensor_hat = cp_combine(factor)
        var_alpha = alpha + 0.5 * sparse_tensor[pos_train].shape[0]
        var_beta = beta + 0.5 * np.sum((sparse_tensor - tensor_hat)[pos_train] ** 2)
        tau = np.random.gamma(var_alpha, 1 / var_beta)
        if it + 1 > burnin_iter:
            factor_plus = [factor_plus[k] + factor[k] for k in range(len(dim))]
            tensor_hat_plus += tensor_hat
        if (it + 1) % 200 == 0 and it < burnin_iter:
            print('Iter: {}'.format(it + 1))
            print('RMSE: {:.6}'.format(Compute_RMSE(dense_tensor[pos_test], tensor_hat[pos_test])))
            print()

    factor = [i / gibbs_iter for i in factor_plus]
    tensor_hat = tensor_hat_plus / gibbs_iter
    print('Final MAPE: {:.6}'.format(Compute_MAPE(dense_tensor[pos_test], tensor_hat[pos_test])))
    print('Final RMSE: {:.6}'.format(Compute_RMSE(dense_tensor[pos_test], tensor_hat[pos_test])))
    print()

    return tensor_hat, factor