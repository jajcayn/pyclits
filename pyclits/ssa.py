"""
Singular Spectrum Analysis

(c) Nikola Jajcay
"""


import numpy as np


class SSA:
    """
    Holds data and performs M-SSA.
    Can perform rotated M-SSA.
    """

    def __init__(self, X, M, compute_rc=True):
        """

        :param X: input data matrix, (time x dim)
        :type X: np.ndarray
        :param M: embedding window
        :type M: int
        """

        if X.ndim == 1:
            X = np.atleast_2d(X).T
        self.X = X
        self.n, self.d = X.shape
        self.M = M
        self.compute_rc = compute_rc
        self.rotated = False

    @staticmethod
    def _shift(arr, n, order="forward"):
        """
        Helper function for time embedding.
        """

        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        if order == "forward":
            shifted = arr[n:] + [0] * n
        elif order == "reversed":
            shifted = [0] * n + arr[:-n]
        else:
            raise ValueError("Unknown order")

        return shifted

    def _get_cov_matrix(self, x, rank_def=False):
        """
        Helper function to obtain structured cov. matrix from X.
        Standardises data, embeds into M-dimensions and computes C.

        :param x: input matrix
        :type x: np.ndarray
        :param rank_def: whether to correct for rank deficiency
        :type rank_def: bool
        """

        # center and normalise
        x = x - np.mean(x, axis=0)
        x = x / np.std(x, axis=0, ddof=1)

        # embed
        aug_x = np.zeros((self.n, self.d * self.M))
        for ch in range(self.d):
            tmp = np.c_[[self._shift(x[:, ch], m) for m in range(self.M)]].T
            aug_x[:, ch * self.M : (ch + 1) * self.M] = tmp

        # cov matrix
        if not rank_def:
            C = np.dot(aug_x.T, aug_x) / self.n
        else:
            C = np.dot(aug_x, aug_x.T) / (self.d * self.M)

        return C, aug_x

    def run(self):
        """
        Performs multichannel SSA (M-SSA) on data matrix X. For ease of
        computing reconstructed components, the PC time series have full length
        of N, not N-M+1.

        Groth & Ghil, Physical Review E, 84(3), 2011.

        :return: eigenvalues (M x D), eigenvectors (M*D x M*D), principal
            components (N x M*D), and reconstructed components (N x D x M)
        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        # cov matrix
        self.C, aug_x = self._get_cov_matrix(self.X)

        # eigendecomposition
        # diag(lambda) = E.T * C * E
        # lambda are eigenvalues, cols of E are eigenvectors
        _, lam, e = np.linalg.svd(self.C, compute_uv=True)
        e = e.T
        assert np.allclose(np.diag(lam), np.dot(e.T, np.dot(self.C, e)))

        self.lambda_sum = np.sum(lam)
        lam /= self.lambda_sum
        ndx = lam.argsort()[::-1]
        self.e = e[:, ndx]  # d*M x d*M
        self.lam = lam[ndx]

        # principal components
        self.pc = np.dot(aug_x, e)

        # reconstructed components
        if self.compute_rc:
            self.rc = np.zeros((self.n, self.d, self.d * self.M))
            for ch in range(self.d):
                for m in np.arange(self.d * self.M):
                    Z = np.zeros(
                        (self.n, self.M)
                    )  # Time-delayed embedding of PC[:, m].
                    for m2 in np.arange(self.M):
                        Z[m2 - self.n :, m2] = self.pc[: self.n - m2, m]

                    # Determine RC as a scalar product.
                    self.rc[:, ch, m] = np.dot(
                        Z, self.e[ch * self.M : (ch + 1) * self.M, m] / self.M
                    )

            return self.lam, self.e, self.pc, np.squeeze(self.rc)
        else:
            return self.lam, self.e, self.pc

    def apply_varimax(self, num_eigen, structured=True, sort_lam=False):
        """
        Performs VARIMAX rotation on M-SSA eigenvectors.

        According to Portes & Aguirre, Physical Review E, 93(5), 2016.

        :param num_eigen: number of eigenvectors entering the rotation
        :type num_eigen: int
        :param structured: whether to apply structured VARIMAX or basic
        :type structured: bool
        :return: eigenvalues (M x D), rotated eigenvectors (M*D x M*D),
            principal components (N x M*D), and reconstructed components
            (N x D x M)
        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """

        self.num_eigen = num_eigen
        self.rotated = True
        if structured:
            rot_matrix = self._get_structured_varimax_rotation_matrix()
        else:
            rot_matrix = self._get_orthomax_rotation_matrix()

        # rotated eigenvectors
        self.Es_rot = np.dot(self.e[:, : self.num_eigen], rot_matrix)

        # rotated eigenvalues
        m_lam = np.diag(self.lam[: self.num_eigen])
        self.lam_rot = np.diag(np.dot(rot_matrix.T, np.dot(m_lam, rot_matrix)))
        if sort_lam:
            ndx = self.lam_rot.argsort()[::-1]
            self.lam_rot = self.lam_rot[ndx]
            self.Es_rot = self.Es_rot[:, ndx]

        # rotated PCs
        self.pc_rot = np.dot(self.pc[:, : self.num_eigen], rot_matrix)

        # rotated RCs
        if self.compute_rc:
            self.rc_rot = np.zeros((self.n, self.d, self.d * self.M))
            pc_mix = self.pc.copy()
            pc_mix[:, : self.num_eigen] = self.pc_rot.copy()
            e_mix = self.e.copy()
            e_mix[:, : self.num_eigen] = self.Es_rot.copy()
            for ch in range(self.d):
                for m in np.arange(self.d * self.M):
                    Z = np.zeros(
                        (self.n, self.M)
                    )  # Time-delayed embedding of PC[:, m].
                    for m2 in np.arange(self.M):
                        Z[m2 - self.n :, m2] = pc_mix[: self.n - m2, m]

                    # Determine RC as a scalar product.
                    self.rc_rot[:, ch, m] = np.dot(
                        Z, e_mix[ch * self.M : (ch + 1) * self.M, m] / self.M
                    )

            return (
                self.lam_rot,
                self.Es_rot,
                self.pc_rot,
                np.squeeze(self.rc_rot),
            )
        else:
            return self.lam_rot, self.Es_rot, self.pc_rot

    def _get_structured_varimax_rotation_matrix(
        self, gamma=1.0, max_iter=20, tol=1e-6
    ):
        """
        Computes the structured rotation matrix.

        Portes & Aguirre, Physical Review E, 93(5), 2016.

        :param gamma: rotation coefficient, gamma=1.0 represents default VARIMAX
        :type gamma: float
        :param max_iter: maximum number of iterations
        :type max_iter: int
        :param tol: tolerance for convergence
        :type tol: float
        :return: structured rotation matrix
        :rtype: np.ndarray
        """

        Ascaled = (self.lam[: self.num_eigen] ** 2) * self.e[
            :, : self.num_eigen
        ]

        _, k = Ascaled.shape
        T, d = np.eye(k), 0

        vec_i = np.array(self.M * [1]).reshape((1, self.M))
        I_d = np.eye(self.d)
        I_d_md = np.kron(I_d, vec_i)
        M = I_d - (gamma / self.d) * np.ones((self.d, self.d))
        IMI = np.dot(I_d_md.T, np.dot(M, I_d_md))

        for _ in range(max_iter):
            d_old = d
            B = np.dot(Ascaled, T)
            G = np.dot(Ascaled.T, B * np.dot(IMI, B ** 2))
            u, s, vh = np.linalg.svd(G)
            T = np.dot(u, vh)
            d = sum(s)
            if d_old != 0 and d / d_old < 1 + tol:
                break

        return T

    def _get_orthomax_rotation_matrix(self, gamma=1.0, max_iter=20, tol=1e-6):
        """
        Computes the orthomax rotation matrix.

        Portes & Aguirre, Physical Review E, 93(5), 2016.

        :param gamma: rotation coefficient, gamma=1.0 represents default VARIMAX
        :type gamma: float
        :param max_iter: maximum number of iterations
        :type max_iter: int
        :param tol: tolerance for convergence
        :type tol: float
        :return: orthomax rotation matrix
        :rtype: np.ndarray
        """

        Ascaled = (self.lam[: self.num_eigen] ** 2) * self.e[
            :, : self.num_eigen
        ]

        p, k = Ascaled.shape
        R, d = np.eye(k), 0

        for _ in range(max_iter):
            d_old = d
            Lambda = np.dot(Ascaled, R)
            u, s, vh = np.linalg.svd(
                np.dot(
                    Ascaled.T,
                    np.asarray(Lambda) ** 3
                    - (gamma / p)
                    * np.dot(
                        Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda)))
                    ),
                )
            )
            R = np.dot(u, vh)
            d = np.sum(s)
            if d_old != 0 and d / d_old < 1 + tol:
                break

        return R
