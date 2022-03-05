import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from scipy.stats import wishart
from scipy.stats import invwishart
from numpy.linalg import inv as inv


def kr_prod(a, b):
    return np.einsum('ir, jr -> ijr', a, b).reshape(a.shape[0] * b.shape[0], -1)
def cov_mat(mat):
    dim1, dim2 = mat.shape
    new_mat = np.zeros((dim2, dim2))
    mat_bar = np.mean(mat, axis = 0)
    for i in range(dim1):
        new_mat += np.einsum('i, j -> ij', mat[i, :] - mat_bar, mat[i, :] - mat_bar)
    return new_mat
def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')
def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)
def mnrnd(M, U, V):
    """
    Generate matrix normal distributed random matrix.
    M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
    """
    dim1, dim2 = M.shape
    X0 = np.random.rand(dim1, dim2)
    P = np.linalg.cholesky(U)
    Q = np.linalg.cholesky(V)
    return M + np.matmul(np.matmul(P, X0), Q.T)


def BTMF(dense_mat, sparse_mat, init, rank, time_lags, maxiter1, maxiter2, file):
    """Bayesian Temporal Matrix Factorization, BTMF."""
    W = init["W"]
    X = init["X"]

    d = time_lags.shape[0]
    dim1, dim2 = sparse_mat.shape
    pos = np.where((dense_mat != 0) & (sparse_mat == 0))
    position = np.where(sparse_mat != 0)
    binary_mat = np.zeros((dim1, dim2))
    binary_mat[position] = 1

    beta0 = 1
    nu0 = rank
    mu0 = np.zeros((rank))
    W0 = np.eye(rank)
    tau = 1
    alpha = 1e-6
    beta = 1e-6
    S0 = np.eye(rank)
    Psi0 = np.eye(rank * d)
    M0 = np.zeros((rank * d, rank))

    W_plus = np.zeros((dim1, rank))
    X_plus = np.zeros((dim2, rank))
    X_new_plus = np.zeros((dim2 + 1, rank))
    A_plus = np.zeros((rank, rank, d))
    mat_hat_plus = np.zeros((dim1, dim2 + 1))

    start_time = time.time()

    for iters in range(maxiter1):
        W_bar = np.mean(W, axis=0)
        var_mu_hyper = (dim1 * W_bar) / (dim1 + beta0)
        var_W_hyper = inv(inv(W0) + cov_mat(W) + dim1 * beta0 / (dim1 + beta0) * np.outer(W_bar, W_bar))
        var_Lambda_hyper = wishart(df=dim1 + nu0, scale=var_W_hyper, seed=None).rvs()
        var_mu_hyper = mvnrnd(var_mu_hyper, inv((dim1 + beta0) * var_Lambda_hyper))

        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = tau * np.matmul(var2, binary_mat.T).reshape([rank, rank, dim1]) + np.dstack([var_Lambda_hyper] * dim1)
        var4 = (tau * np.matmul(var1, sparse_mat.T)
                + np.dstack([np.matmul(var_Lambda_hyper, var_mu_hyper)] * dim1)[0, :, :])
        for i in range(dim1):
            inv_var_Lambda = inv(var3[:, :, i])
            W[i, :] = mvnrnd(np.matmul(inv_var_Lambda, var4[:, i]), inv_var_Lambda)
        if iters + 1 > maxiter1 - maxiter2:
            W_plus += W

        Z_mat = X[np.max(time_lags): dim2, :]
        Q_mat = np.zeros((dim2 - np.max(time_lags), rank * d))
        for t in range(np.max(time_lags), dim2):
            Q_mat[t - np.max(time_lags), :] = X[t - time_lags, :].reshape([rank * d])
        var_Psi = inv(inv(Psi0) + np.matmul(Q_mat.T, Q_mat))
        var_M = np.matmul(var_Psi, np.matmul(inv(Psi0), M0) + np.matmul(Q_mat.T, Z_mat))
        var_S = (S0 + np.matmul(Z_mat.T, Z_mat) + np.matmul(np.matmul(M0.T, inv(Psi0)), M0)
                 - np.matmul(np.matmul(var_M.T, inv(var_Psi)), var_M))
        Sigma = invwishart(df=nu0 + dim2 - np.max(time_lags), scale=var_S, seed=None).rvs()
        A = mat2ten(mnrnd(var_M, var_Psi, Sigma).T, np.array([rank, rank, d]), 0)
        if iters + 1 > maxiter1 - maxiter2:
            A_plus += A

        Lambda_x = inv(Sigma)
        var1 = W.T
        var2 = kr_prod(var1, var1)
        var3 = tau * np.matmul(var2, binary_mat).reshape([rank, rank, dim2]) + np.dstack([Lambda_x] * dim2)
        var4 = tau * np.matmul(var1, sparse_mat)
        for t in range(dim2):
            Mt = np.zeros((rank, rank))
            Nt = np.zeros(rank)
            if t < np.max(time_lags):
                Qt = np.zeros(rank)
            else:
                Qt = np.matmul(Lambda_x, np.matmul(ten2mat(A, 0), X[t - time_lags, :].reshape([rank * d])))
            if t < dim2 - np.min(time_lags):
                if t >= np.max(time_lags) and t < dim2 - np.max(time_lags):
                    index = list(range(0, d))
                else:
                    index = list(np.where((t + time_lags >= np.max(time_lags)) & (t + time_lags < dim2)))[0]
                for k in index:
                    Ak = A[:, :, k]
                    Mt += np.matmul(np.matmul(Ak.T, Lambda_x), Ak)
                    A0 = A.copy()
                    A0[:, :, k] = 0
                    var5 = (X[t + time_lags[k], :]
                            - np.matmul(ten2mat(A0, 0), X[t + time_lags[k] - time_lags, :].reshape([rank * d])))
                    Nt += np.matmul(np.matmul(Ak.T, Lambda_x), var5)
            var_mu = var4[:, t] + Nt + Qt
            if t < np.max(time_lags):
                inv_var_Lambda = inv(var3[:, :, t] + Mt - Lambda_x + np.eye(rank))
            else:
                inv_var_Lambda = inv(var3[:, :, t] + Mt)
            X[t, :] = mvnrnd(np.matmul(inv_var_Lambda, var_mu), inv_var_Lambda)
        mat_hat = np.matmul(W, X.T)

        X_new = np.zeros((dim2 + 1, rank))
        if iters + 1 > maxiter1 - maxiter2:
            X_new[0: dim2, :] = X.copy()
            X_new[dim2, :] = np.matmul(ten2mat(A, 0), X_new[dim2 - time_lags, :].reshape([rank * d]))
            X_new_plus += X_new
            mat_hat_plus += np.matmul(W, X_new.T)

        tau = np.random.gamma(alpha + 0.5 * sparse_mat[position].shape[0],
                              1 / (beta + 0.5 * np.sum((sparse_mat - mat_hat)[position] ** 2)))
        rmse = np.sqrt(np.sum((dense_mat[pos] - mat_hat[pos]) ** 2) / dense_mat[pos].shape[0])
        if (iters + 1) % 200 == 0 and iters < maxiter1 - maxiter2:
            #print('Iter: {}'.format(iters + 1))
            print('MAE: {:.6}'.format(
                np.sum(np.abs(dense_mat[pos] - mat_hat[pos])) / dense_mat[pos].shape[0]))
            print('MAPE: {:.6}'.format(np.sum(np.abs(dense_mat[pos] - mat_hat[pos]) / dense_mat[pos]) / dense_mat[pos].shape[0]))
            #print('time consume : {:.6}'.format(time.time() - start_time))
        #print('epoch {}, time consume :{}'.format(iters, time.time() - start_time))
    W = W_plus / maxiter2
    X_new = X_new_plus / maxiter2
    A = A_plus / maxiter2
    mat_hat = mat_hat_plus / maxiter2
    if maxiter1 >= 100:
        final_mae = np.sum(np.abs(dense_mat[pos] - mat_hat[pos])) / dense_mat[pos].shape[0]
        final_mape = np.sum(np.abs(dense_mat[pos] - mat_hat[pos]) / dense_mat[pos]) / dense_mat[pos].shape[0]
        final_mse = np.sum((dense_mat[pos] - mat_hat[pos]) ** 2) / dense_mat[pos].shape[0]
        with open(file, "w") as fl:
            fl.write('Imputation MAE: {:.6}\n'.format(final_mae))
            fl.write('Imputation MAPE: {:.6}\n'.format(final_mape))
            fl.write('Imputation MSE: {:.6}\n'.format(final_mse))
            fl.write('TIME consume: {:.2}\n'.format(time.time() - start_time))

    return mat_hat, W, X_new, A
# def mvnrnd_pre(mu, Lambda):
#     src = normrnd(size = (mu.shape[0],))
#     return solve_ut(cholesky_upper(Lambda, overwrite_a = True, check_finite = False),
#                     src, lower = False, check_finite = False, overwrite_b = True) + mu
#
# def cov_mat(mat, mat_bar):
#     mat = mat - mat_bar
#     return mat.T @ mat
#
#
# def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
#     """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""
#
#     dim1, rank = W.shape
#     W_bar = np.mean(W, axis=0)
#     temp = dim1 / (dim1 + beta0)
#     var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
#     var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
#     var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)
#
#     if dim1 * rank ** 2 > 1e+8:
#         vargin = 1
#
#     if vargin == 0:
#         var1 = X.T
#         var2 = kr_prod(var1, var1)
#         var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
#         var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
#         for i in range(dim1):
#             W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
#     elif vargin == 1:
#         for i in range(dim1):
#             pos0 = np.where(sparse_mat[i, :] != 0)
#             Xt = X[pos0[0], :]
#             var_mu = tau[i] * Xt.T @ sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
#             var_Lambda = tau[i] * Xt.T @ Xt + var_Lambda_hyper
#             W[i, :] = mvnrnd_pre(solve(var_Lambda, var_mu), var_Lambda)
#
#     return W
#
#
# def mnrnd(M, U, V):
#     """
#     Generate matrix normal distributed random matrix.
#     M is a m-by-n matrix, U is a m-by-m matrix, and V is a n-by-n matrix.
#     """
#     dim1, dim2 = M.shape
#     X0 = np.random.randn(dim1, dim2)
#     P = cholesky_lower(U)
#     Q = cholesky_lower(V)
#
#     return M + P @ X0 @ Q.T
#
#
# def sample_var_coefficient(X, time_lags):
#     dim, rank = X.shape
#     d = time_lags.shape[0]
#     tmax = np.max(time_lags)
#
#     Z_mat = X[tmax: dim, :]
#     Q_mat = np.zeros((dim - tmax, rank * d))
#     for k in range(d):
#         Q_mat[:, k * rank: (k + 1) * rank] = X[tmax - time_lags[k]: dim - time_lags[k], :]
#     var_Psi0 = np.eye(rank * d) + Q_mat.T @ Q_mat
#     var_Psi = inv(var_Psi0)
#     var_M = var_Psi @ Q_mat.T @ Z_mat
#     var_S = np.eye(rank) + Z_mat.T @ Z_mat - var_M.T @ var_Psi0 @ var_M
#     Sigma = invwishart.rvs(df=rank + dim - tmax, scale=var_S)
#
#     return mnrnd(var_M, var_Psi, Sigma), Sigma
#
#
# def sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, Lambda_x):
#     """Sampling T-by-R factor matrix X."""
#
#     dim2, rank = X.shape
#     tmax = np.max(time_lags)
#     tmin = np.min(time_lags)
#     d = time_lags.shape[0]
#     A0 = np.dstack([A] * d)
#     for k in range(d):
#         A0[k * rank: (k + 1) * rank, :, k] = 0
#     mat0 = Lambda_x @ A.T
#     mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
#     mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))
#
#     var1 = W.T
#     var2 = kr_prod(var1, var1)
#     var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]
#     var4 = var1 @ tau_sparse_mat
#     for t in range(dim2):
#         Mt = np.zeros((rank, rank))
#         Nt = np.zeros(rank)
#         Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
#         index = list(range(0, d))
#         if t >= dim2 - tmax and t < dim2 - tmin:
#             index = list(np.where(t + time_lags < dim2))[0]
#         elif t < tmax:
#             Qt = np.zeros(rank)
#             index = list(np.where(t + time_lags >= tmax))[0]
#         if t < dim2 - tmin:
#             Mt = mat2.copy()
#             temp = np.zeros((rank * d, len(index)))
#             n = 0
#             for k in index:
#                 temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
#                 n += 1
#             temp0 = X[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
#             Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)
#
#         var3[:, :, t] = var3[:, :, t] + Mt
#         if t < tmax:
#             var3[:, :, t] = var3[:, :, t] - Lambda_x + np.eye(rank)
#         X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])
#
#     return X
#
# def sample_precision_tau(sparse_mat, mat_hat, ind):
#     var_alpha = 1e-6 + 0.5 * np.sum(ind, axis = 1)
#     var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis = 1)
#     return np.random.gamma(var_alpha, 1 / var_beta)
#
# def sample_precision_scalar_tau(sparse_mat, mat_hat, ind):
#     var_alpha = 1e-6 + 0.5 * np.sum(ind)
#     var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind)
#     return np.random.gamma(var_alpha, 1 / var_beta)
#
# def compute_mae(var, var_hat):
#     return np.sum(np.abs(var - var_hat)) / var.shape[0]
#
# def compute_mape(var, var_hat):
#     return np.sum(np.abs(var - var_hat) / var) / var.shape[0]
#
# # def compute_rmse(var, var_hat):
# #     return  np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
# def compute_mse(var, var_hat):
#     return  np.sum((var - var_hat) ** 2) / var.shape[0]
#
#
# def BTMF(dense_mat, sparse_mat, init, rank, time_lags, burn_iter, gibbs_iter, file, option="factor"):
#     """Bayesian Temporal Matrix Factorization, BTMF."""
#     # print('dense mat shape {}'.format(dense_mat.shape))
#     dim1, dim2 = sparse_mat.shape
#     d = time_lags.shape[0]
#     W = init["W"]
#     X = init["X"]
#     if np.isnan(sparse_mat).any() == False:
#         ind = sparse_mat != 0
#         pos_obs = np.where(ind)
#         pos_test = np.where((dense_mat != 0) & (sparse_mat == 0))
#     elif np.isnan(sparse_mat).any() == True:
#         pos_test = np.where((dense_mat != 0) & (np.isnan(sparse_mat)))
#         ind = ~np.isnan(sparse_mat)
#         pos_obs = np.where(ind)
#         sparse_mat[np.isnan(sparse_mat)] = 0
#
#     #print('dense mat shape {}'.format(dense_mat.shape))
#     dense_test = dense_mat[pos_test]
#
#     #print('dense_test shape :{}'.format(dense_test.shape))
#     del dense_mat
#     tau = np.ones(dim1)
#     W_plus = np.zeros((dim1, rank))
#     X_plus = np.zeros((dim2, rank))
#     A_plus = np.zeros((rank * d, rank))
#     temp_hat = np.zeros(len(pos_test[0]))
#     show_iter = 200
#     mat_hat_plus = np.zeros((dim1, dim2))
#     for it in range(burn_iter + gibbs_iter):
#         start_time = time.time()
#
#         tau_ind = tau[:, None] * ind
#         tau_sparse_mat = tau[:, None] * sparse_mat
#         W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau)
#         A, Sigma = sample_var_coefficient(X, time_lags)
#         X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, A, inv(Sigma))
#         mat_hat = W @ X.T
#         if option == "factor":
#             tau = sample_precision_tau(sparse_mat, mat_hat, ind)
#         elif option == "pca":
#             tau = sample_precision_scalar_tau(sparse_mat, mat_hat, ind)
#             tau = tau * np.ones(dim1)
#         temp_hat += mat_hat[pos_test]
#         if (it + 1) % show_iter == 0 and it < burn_iter:
#             temp_hat = temp_hat / show_iter
#             # print('temp_hat shape :{}'.format(temp_hat.shape))
#             # print('Iter: {}'.format(it + 1))
#             # print('MAE: {:.6}'.format(compute_mae(dense_test, temp_hat)))
#             # print('MAPE: {:.6}'.format(compute_mape(dense_test, temp_hat)))
#             # print('RMSE: {:.6}'.format(compute_rmse(dense_test, temp_hat)))
#             # print("time consume : {}".format(time.time() - start_time))
#             temp_hat = np.zeros(len(pos_test[0]))
#             print()
#         if it + 1 > burn_iter:
#             W_plus += W
#             X_plus += X
#             A_plus += A
#             mat_hat_plus += mat_hat
#     mat_hat = mat_hat_plus / gibbs_iter
#     W = W_plus / gibbs_iter
#     X = X_plus / gibbs_iter
#     A = A_plus / gibbs_iter
#
#     # print('Imputation MAE: {:.6}'.format(compute_mae(dense_test, mat_hat[:, : dim2][pos_test])))
#     # print('Imputation MAPE: {:.6}'.format(compute_mape(dense_test, mat_hat[:, : dim2][pos_test])))
#     # print('Imputation RMSE: {:.6}'.format(compute_rmse(dense_test, mat_hat[:, : dim2][pos_test])))
#     # print()
#
#     with open(file, "a") as fl:
#         fl.write('Imputation MAE: {:.6}\n'.format(compute_mae(dense_test, mat_hat[:, : dim2][pos_test])))
#         fl.write('Imputation MAPE: {:.6}\n'.format(compute_mape(dense_test, mat_hat[:, : dim2][pos_test])))
#         fl.write('Imputation MSE: {:.6}\n'.format(compute_mse(dense_test, mat_hat[:, : dim2][pos_test])))
#
#     mat_hat[mat_hat < 0] = 0
#
#     return mat_hat, W, X, A



import time

if __name__ == '__main__':
    count = 0
    while(1):
      print(count)
      time.sleep(2)
      count = count + 1
      with open("./reuslt", "a") as f:
          f.write("" + count)
