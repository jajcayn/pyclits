"""
created on Aug 21, 2016

@author: Nikola Jajcay, jajcay(at)cs.cas.cz

last update on Sep 22, 2017
"""


import numpy as np


class ssa_class():
    """
    Holds data and performs M-SSA.
    Can perform rotated M-SSA.
    """

    def __init__(self, X, M, compute_rc = True):
        """
        X is input data matrix as time x dimension -- N x D.
        If X is univariate, analysis could be performed as well, M-SSA reduces to classic SSA.
        M is embedding window. 
        """

        if X.ndim == 1:
            X = np.atleast_2d(X).T
        self.X = X
        self.n, self.d = X.shape
        self.M = M
        self.compute_rc = compute_rc
        self.T = None
        self.rotated = False



    @staticmethod
    def _shift(arr, n, order = 'forward'):
        """
        Helper function for time embedding. 
        """

        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if order == 'forward':
            shifted = arr[n:] + [0] * n
        elif order == 'reversed':
            shifted = [0] * n + arr[:-n]
        else:
            print("Order %s not recognized.  Try forward or reversed" % order)

        return shifted



    def _get_cov_matrix(self, x, rank_def = False):
        """
        Helper function to obtain structured cov. matrix from X.
        Standardises data, embeds into M-dimensions and computes C.
        """

        # center and normalise
        for i in range(self.d):
            x[:, i] -= np.mean(x[:, i])
            x[:, i] /= np.std(x[:, i], ddof = 1)

        # embed
        aug_x = np.zeros((self.n, self.d*self.M))
        for ch in range(self.d):
            tmp = np.c_[[self._shift(x[:, ch], m) for m in range(self.M)]].T
            aug_x[:, ch*self.M:(ch+1)*self.M] = tmp

        # cov matrix
        if not rank_def:
            C = np.dot(aug_x.T, aug_x) / self.n
        else:
            C = np.dot(aug_x, aug_x.T) / (self.d*self.M)

        return C, aug_x



    def run_ssa(self):
        """
        Performs multichannel SSA (M-SSA) on data matrix X.
        X as time x dimension -- N x D. 
        For simplicity of computing reconstructed components, the 
        PC time series have full length of N, not N-M+1.

        According to Groth & Ghil (2011), Physical Review E, 84(3).

        Return as eigenvalues (M*D), eigenvectors (M*D x M*D), 
        principal components (N x M*D) and recontructed components (N x D x M).
        """

        # cov matrix
        self.C, aug_x = self._get_cov_matrix(self.X)

        # eigendecomposition
        u, lam, e = np.linalg.svd(self.C, compute_uv = True) # diag(lambda) = E.T * C * E, lambda are eigenvalues, cols of E are eigenvectors
        e = e.T
        assert np.allclose(np.diag(lam), np.dot(e.T, np.dot(self.C, e)))
        
        self.lambda_sum = np.sum(lam)
        lam /= self.lambda_sum
        ndx = lam.argsort()[::-1]
        self.e = e[:, ndx] # d*M x d*M
        self.lam = lam[ndx]

        # principal components
        self.pc = np.dot(aug_x, e)

        # reconstructed components
        if self.compute_rc:
            self.rc = np.zeros((self.n, self.d, self.d*self.M))
            for ch in range(self.d):
                for m in np.arange(self.d*self.M):
                    Z = np.zeros((self.n, self.M))  # Time-delayed embedding of PC[:, m].
                    for m2 in np.arange(self.M):
                        Z[m2 - self.n:, m2] = self.pc[:self.n - m2, m]

                    # Determine RC as a scalar product.
                    self.rc[:, ch, m] = np.dot(Z, self.e[ch*self.M:(ch+1)*self.M, m] / self.M)


            return self.lam, self.e, self.pc, np.squeeze(self.rc)
        else:
            return self.lam, self.e, self.pc



    def _get_structured_varimax_rotation_matrix(self, gamma = 1., q = 20, tol = 1e-6):
        """
        Computes the rotation matrix T.
        S is number of eigenvectors entering the rotation

        Adapted from Portes & Aguirre (2016), Physical Review E, 93(5).

        Returns rotation matrix T.
        """

        Ascaled = (self.lam[:self.S]**2) * self.e[:, :self.S]

        p, k = Ascaled.shape
        T, d = np.eye(k), 0

        vec_i = np.array(self.M*[1]).reshape((1, self.M))
        I_d = np.eye(self.d)
        I_d_md = np.kron(I_d, vec_i)
        M = I_d - (gamma/self.d) * np.ones((self.d, self.d))
        IMI = np.dot(I_d_md.T, np.dot(M, I_d_md))

        for i in range(q):
            d_old = d
            B = np.dot(Ascaled, T)
            G = np.dot(Ascaled.T, B * np.dot(IMI, B**2))
            u, s, vh = np.linalg.svd(G)
            T = np.dot(u, vh)
            d = sum(s)
            if d_old != 0 and d/d_old < 1 + tol:
                break

        # T is rotation matrix
        self.T = T



    def _get_orthomax_rotation_matrix(self, gamma = 1.0, q = 20, tol = 1e-6):
        """
        Computes the rotation matrix T.
        S is number of eigenvectors entering the rotation

        Adapted from Portes & Aguirre (2016), Physical Review E, 93(5).

        Returns rotation matrix T.
        """

        Ascaled = (self.lam[:self.S]**2) * self.e[:, :self.S]

        p, k = Ascaled.shape
        R, d = np.eye(k), 0

        for i in range(q):
            d_old = d
            Lambda = np.dot(Ascaled, R)
            u, s, vh = np.linalg.svd(np.dot(Ascaled.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(u,vh)
            d = np.sum(s)
            if d_old != 0 and d/d_old < 1 + tol: 
                break
        
        # R is rotation matrix
        self.T = R



    def apply_varimax(self, S, structured = True, sort_lam = False):
        """
        Performs varimax rotation on M-SSA eigenvectors.
        S is number of eigenvectors entering the rotation.
        If structured is True, applies structured varimax rotation, if False applies basic orthomax.
        
        According to Portes & Aguirre (2016), Physical Review E, 93(5).

        Returns as M-SSA, but rotated.
        """

        self.S = S
        self.rotated = True
        if structured:
            self._get_structured_varimax_rotation_matrix()
        else:
            self._get_orthomax_rotation_matrix()

        # rotated eigenvectors
        self.Es_rot = np.dot(self.e[:, :self.S], self.T)

        # rotated eigenvalues
        m_lam = np.diag(self.lam[:self.S])
        self.lam_rot = np.diag(np.dot(self.T.T, np.dot(m_lam, self.T)))
        if sort_lam:
            ndx = self.lam_rot.argsort()[::-1]
            self.lam_rot = self.lam_rot[ndx]
            self.Es_rot = self.Es_rot[:, ndx]

        # rotated PCs
        self.pc_rot = np.dot(self.pc[:, :self.S], self.T)

        # rotated RCs
        if self.compute_rc:
            self.rc_rot = np.zeros((self.n, self.d, self.d*self.M))
            pc_mix = self.pc.copy()
            pc_mix[:, :self.S] = self.pc_rot.copy()
            e_mix = self.e.copy()
            e_mix[:, :self.S] = self.Es_rot.copy() 
            for ch in range(self.d):
                for m in np.arange(self.d*self.M):
                    Z = np.zeros((self.n, self.M))  # Time-delayed embedding of PC[:, m].
                    for m2 in np.arange(self.M):
                        Z[m2 - self.n:, m2] = pc_mix[:self.n - m2, m]

                    # Determine RC as a scalar product.
                    self.rc_rot[:, ch, m] = np.dot(Z, e_mix[ch*self.M:(ch+1)*self.M, m] / self.M)

            return self.lam_rot, self.Es_rot, self.pc_rot, np.squeeze(self.rc_rot)
        else:
            return self.lam_rot, self.Es_rot, self.pc_rot



    def _get_MC_realizations(self, n = 100, multivariate = False, residuals = True):
        """
        Gets n surrogates for Monte Carlo testing. 
        If multivariate True, extimates AR(1) model for whole data, if False, treats as univariate
        and estimates each channel separately.
        If residuals True, generates AR model using actual residuals from fitting, if False,
        only uses model matrix A.
        """

        from var_model import VARModel

        self.MCsurrs = np.zeros([n] + list(self.X.shape))
        
        # multivariate model
        if multivariate:
            v = VARModel()
            v.estimate(self.X, [1,1], True, 'sbc', None)
            if residuals:
                r = v.compute_residuals(self.X)
        
        # univariate model - estimating for each channel separately
        else:
            vs = {}
            for d in range(self.X.shape[1]):
                vs[d] = VARModel()
                vs[d].estimate(self.X[:, d], [1,1], True, 'sbc', None)
                if residuals:
                    vs['res' + str(d)] = vs[d].compute_residuals(self.X[:, d])

        for i in range(n):
            if multivariate:
                if not residuals:
                    self.MCsurrs[i, ...] = v.simulate(N = self.X.shape[0])
                else:
                    self.MCsurrs[i, ...] = v.simulate_with_residuals(r, orig_length = True)
            else:
                for d in range(self.X.shape[1]):   
                    if not residuals:
                        self.MCsurrs[i, :, d] = np.squeeze(vs[d].simulate(N = self.X.shape[0]))
                    else:
                        self.MCsurrs[i, :, d] = np.squeeze(vs[d].simulate_with_residuals(vs['res' + str(d)], orig_length = True))



    def run_Monte_Carlo(self, n_realizations, p_value = 0.05, method = 'rotation', multivariate = False, residuals = True, 
            plot = True, return_eigvals = False):
        """
        Performs Monte Carlo SSA.
        Computes n realizations of stochastic AR(1) process fitted onto data.
        method: (to obtain eigenspectrum of AR surrogates)
          'data' - project cov. matrix of surrogates onto data eigenvectors
          'separately' - compute SVD per surrogate
          'ensemble' - average all cov. matrices into one, and project cov. matrices onto this average eigenvectors
          'rotation' - projection is done onto rotation approximation (G&G 2015, J. Clim.)
          'rank-def' - rank deficient method, used when N' < DM
        multivariate (True | False) - whether to fit multivariate model or univariate  
        residuals (True | False) - whether simulate with residuals or just use model matrix A
        plot (True | False) - whether to plot data vs. surr. eigenvalues (for visual inspection) - recommended!
        return_eigvals (True | False) - whether to return all surrogate eigenvalues, for further inspection

        According to Groth & Ghil (2015), Journal of Climate, 28(19). 
        """

        if method not in ['data', 'separately', 'ensemble', 'rotation', 'rank-def']:
            raise Exception("Method not know. Please use one of the 'data', 'separately', 'ensemble', 'rotation', 'rank-def'.")

        # get the ensamble of surrogate data
        self._get_MC_realizations(n = n_realizations, multivariate = multivariate, residuals = residuals)

        if (self.n - self.M + 1) < self.d*self.M:
            print("**WARNING: with the current SSA settings, the cov. matrix is rank deficient. Keep in mind.")

        if method == 'rank-def':
            if (self.n - self.M + 1) > self.d*self.M:
                print("**WARNING: rank deficient method is usually used when N' < DM. Otherwise, 'rotation' method works best.")
            print("**WARNING: in rank deficient matrix the problem is more compicated, rely more on visual spectra than computational method.")
            # cov matrix
            C_rd, _ = self._get_cov_matrix(self.X, rank_def = True)
            # eigendecomposition
            _, l_rd, e_rd = np.linalg.svd(C_rd, compute_uv = True) # diag(lambda) = E.T * C * E, lambda are eigenvalues, cols of E are eigenvectors
            e_rd = e_rd.T
            assert np.allclose(C_rd, np.dot(e_rd, np.dot(np.diag(l_rd), e_rd.T)))
            l_rd = l_rd / np.sum(l_rd)
            ndx = np.argsort(l_rd)[::-1]
            l_rd = l_rd[ndx]

        # if emsemble - first get ensemble cov. matrix and vectors
        if method == 'ensemble':
            C_mean_surr = np.mean(np.array([self._get_cov_matrix(self.MCsurrs[i, ...])[0] for i in range(n_realizations)]), axis = 0)
            _, ltemp, e_mean_surr = np.linalg.svd(C_mean_surr, compute_uv = True) # diag(lambda) = E.T * C * E, lambda are eigenvalues, cols of E are eigenvectors
            e_mean_surr = e_mean_surr.T
            assert np.allclose(C_mean_surr, np.dot(e_mean_surr, np.dot(np.diag(ltemp), e_mean_surr.T)))

            eig_data_ensemble = np.diag(np.dot(e_mean_surr.T, np.dot(self.C, e_mean_surr)))
            eig_data_ensemble = eig_data_ensemble / np.sum(eig_data_ensemble)
            ndx = np.argsort(eig_data_ensemble)[::-1]
            eig_data_ensemble = eig_data_ensemble[ndx]

        eigvals_surrs = []
        for i in range(n_realizations):
            if method == 'rank-def':
                C_surr = self._get_cov_matrix(self.MCsurrs[i, ...], rank_def = True)[0]
            else:
                C_surr = self._get_cov_matrix(self.MCsurrs[i, ...])[0]

            if method == 'data':
                # project on data eigenvectors
                eig_s = np.diag(np.dot(self.e.T, np.dot(C_surr, self.e)))
                # normalise
                eig_s = eig_s / np.sum(eig_s)
                # sort
                ndx = np.argsort(eig_s)[::-1]
                eig_s = eig_s[ndx]
            elif method == 'separately':
                # get eigenvectors for each surrogate separately
                _, eig_s, _ = np.linalg.svd(C_surr, compute_uv = True)
                # normalise
                eig_s /= np.sum(eig_s)
                # sort
                ndx = np.argsort(eig_s)[::-1]
                eig_s = eig_s[ndx]
            elif method == 'ensemble':
                # project on mean eigenvectors from surrs
                eig_s = np.diag(np.dot(e_mean_surr.T, np.dot(C_surr, e_mean_surr)))
                # normalise
                eig_s = eig_s / np.sum(eig_s)
                # sort
                ndx = np.argsort(eig_s)[::-1]
                eig_s = eig_s[ndx]
            elif method == 'rotation' or method == 'rank-def':
                _, eig, e_s = np.linalg.svd(C_surr, compute_uv = True)
                e_s = e_s.T
                if method == 'rotation':
                    u, _, v = np.linalg.svd(np.dot((eig**0.5)*e_s.T, (self.lam**0.5)*self.e))
                elif method == 'rank-def':
                    u, _, v = np.linalg.svd(np.dot((eig**0.5)*e_s.T, (l_rd**0.5)*e_rd))
                T_e = np.dot(u, v)
                # project with rotation
                eig_s = np.diag(np.dot(T_e.T, np.dot(np.diag(eig), T_e)))
                # normalise
                eig_s = eig_s / np.sum(eig_s)
                # sort
                ndx = np.argsort(eig_s)[::-1]
                eig_s = eig_s[ndx]

            eigvals_surrs.append(eig_s)

        eigvals_surrs = np.array(eigvals_surrs)

        if method == 'ensemble':
            data_eigvals = eig_data_ensemble
        elif method == 'rank-def':
            data_eigvals = l_rd
        else:
            data_eigvals = self.lam if not self.rotated else self.lam_rot

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(data_eigvals[:40], marker = 'o', markersize = 10, linestyle = 'none', color = 'k')
            plt.plot(np.percentile(eigvals_surrs[:, :40], q = 97.5, axis = 0), marker = "^", markersize = 6, linestyle = 'none', color = 'g')
            plt.plot(np.percentile(eigvals_surrs[:, :40], q = 2.5, axis = 0), marker = "v", markersize = 6, linestyle = 'none', color = 'g')
            plt.xlabel("ORDER", size = 20)
            plt.title("DATA EIGENVALUES vs. %d MONTE-CARLO AR(1)" % (n_realizations), size = 25)
            plt.show()

        prctl = np.percentile(eigvals_surrs, q = 1 - p_value/2., axis = 0) if not self.rotated else np.percentile(eigvals_surrs[:, :self.S], q = 1 - p_value/2., axis = 0)
        num_sign = 0
        for i in range(data_eigvals.shape[0]):
            if data_eigvals[i] > prctl[i]:
                num_sign += 1
            else:
                break

        if not return_eigvals:
            return num_sign
        else:
            return num, eigvals_surrs



    def run_enhanced_Monte_Carlo(self, n_realizations, to_plot = 60, multivariate = False, residuals = True, return_eigvals = False):
        """
        Performs Enhanced Monte Carlo SSA.

        According to Palus & Novotna (2004), Nonlinear Processes in Geophys., 11(5/6).
        """

        # get the ensamble of surrogate data
        self._get_MC_realizations(n = n_realizations, multivariate = multivariate, residuals = residuals)

        # unlike MC-SSA, we are plotting eigenspectrum not against the order (size), but rather against the 
        # dominant frequency associated with it

        # create bins
        freq_bins = np.linspace(0, 0.5, self.M+1)

        # get freqs from data's PCs
        freqs = np.fft.fftfreq(self.pc.shape[0])
        # sort according to bins
        lam_freqs = np.digitize(np.abs([freqs[np.fft.fft(self.pc[:, i]).argmax()] for i in range(self.pc.shape[1])]), freq_bins)

        # surrogates
        lam_freqs_surrs = np.zeros(([n_realizations] + list(lam_freqs.shape)))
        eigvals_surrs = []
        for i in range(n_realizations):
            # for each surrogate separately
            C_surr, x_surr = self._get_cov_matrix(self.MCsurrs[i, ...])
            # get eigenvectors for each surrogate separately
            _, eig_s, e_surr = np.linalg.svd(C_surr, compute_uv = True)
            e_surr = e_surr.T
            # normalise
            eig_s /= np.sum(eig_s)
            # sort
            ndx = np.argsort(eig_s)[::-1]
            eig_s = eig_s[ndx]
            e_surr = e_surr[:, ndx]
            # PCs
            pc_surr = np.dot(x_surr, e_surr)
            assert pc_surr.shape == self.pc.shape
            # sort to bins
            lam_freqs_surrs[i, ...] = np.digitize(np.abs([freqs[np.fft.fft(pc_surr[:, ii]).argmax()] for ii in range(pc_surr.shape[1])]), freq_bins)

            eigvals_surrs.append(eig_s)

        eigvals_surrs = np.array(eigvals_surrs)

        # plot
        import matplotlib.pyplot as plt
        for ii in range(to_plot):
            plt.plot(freq_bins[lam_freqs[ii]] + np.diff(freq_bins)[0]/2., self.lam[ii], marker = 'o', markersize = 10, linestyle = 'none', color = 'k')
            # plt.plot(freq_bins[int(lam_freqs_surrs[:, ii].mean())] + np.diff(freq_bins)[0]/2., np.percentile(eigvals_surrs[:, ii], q = 97.5, axis = 0),
            #     marker = "^", markersize = 6, linestyle = 'none', color = 'g') 
            # plt.plot(freq_bins[int(lam_freqs_surrs[:, ii].mean())] + np.diff(freq_bins)[0]/2., np.percentile(eigvals_surrs[:, ii], q = 2.5, axis = 0),
            #     marker = "v", markersize = 6, linestyle = 'none', color = 'g') 

        plt.show()

