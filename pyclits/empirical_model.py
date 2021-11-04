"""
Empirical multi-level modelling approach.

Kondrashov D., Kravtsov S., Robertson A. W. and Ghil M., J. Climate, 18, 2005.

(c) Nikola Jajcay
"""
import logging
from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np
import scipy.stats as sts
import xarray as xr
from pathos.multiprocessing import ProcessingPool
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, LinearRegression, Ridge

from functions import cross_correlation, kdensity_estimate

from .geofield import DataField

CCORR_MAX_LAG = 50


def _partial_least_squares(x, y, ux, sx, vx, optimal, intercept=True):
    """
    Partial least squares for regression with regularization.
    """

    xm = np.mean(x, axis=0)
    ym = np.mean(y, axis=0)

    e = ux * sx
    f = y - ym

    w = np.zeros((x.shape[1], optimal))
    if len(y.shape) == 1:
        yshp = 1
    else:
        yshp = y.shape[1]
    c = np.zeros((yshp, optimal))
    t = np.zeros((x.shape[0], optimal))
    u = np.zeros_like(t)
    b = np.zeros((optimal, optimal))
    p = np.zeros_like(w)

    for j in range(optimal):
        l = np.dot(e.T, f)
        if len(l.shape) == 1:
            l = l[:, np.newaxis]
        rv, _, lv = np.linalg.svd(l, full_matrices=False)
        rv, lv = rv[:, 0], np.squeeze(lv.T[:, 0])

        w[:, j] = rv
        c[:, j] = lv
        t[:, j] = np.dot(e, w[:, j])
        t[:, j] /= np.sqrt(np.dot(t[:, j].T, t[:, j]))
        u[:, j] = np.dot(f, np.squeeze(c[:, j]))
        b[j, j] = np.dot(t[:, j].T, u[:, j])
        p[:, j] = np.dot(e.T, t[:, j])

        e -= np.outer(t[:, j], p[:, j].T)
        f -= np.squeeze(
            np.dot(b[j, j], np.outer(t[:, j], np.squeeze(c[:, j].T)))
        )

    bpls1 = np.dot(
        np.dot(np.linalg.pinv(p[:, :optimal].T), b[:optimal, :optimal]),
        np.squeeze(c[:, :optimal].T),
    )
    bpls2 = np.dot(vx[:, : sx.shape[0]], bpls1)

    if intercept:
        # bpls = np.zeros((bpls2.shape[0] + 1, bpls2.shape[1]))
        # bpls[:-1, :] = bpls2
        # bpls[-1, :] = ym - np.dot(xm,bpls2)
        bpls = np.append(bpls2, [ym - np.dot(xm, bpls2)])
        xx = np.c_[x, np.ones(x.shape[0])]
        r = y - np.dot(xx, bpls)
    else:
        bpls = bpls2
        r = y - np.dot(x, bpls)

    return bpls, r


def compute_statistics(pcs, max_lag=CCORR_MAX_LAG):
    """
    Compute statistics (autocorrelation, kernel density estimation, and
    first 4 statistical moments) of input PCs.

    :param pcs: principal components to compute statistics on, (time x dim)
    :type pcs: np.ndarray
    :param max_lag: maximum lag for autocorrelation
    :type max_lag: int
    :return: autocorrelation, kernel density, and statistical moments
    :rtype: xr.DataArray, xr.DataArray, xr.DataArray
    """
    lag_cors = np.zeros((2 * max_lag + 1, pcs.shape[1]))
    kernel_densities = np.zeros((100, pcs.shape[1], 2))
    for k in range(pcs.shape[1]):
        lag_cors[:, k] = cross_correlation(
            pcs[:, k], pcs[:, k], max_lag=max_lag
        )
        (
            kernel_densities[:, k, 0],
            kernel_densities[:, k, 1],
        ) = kdensity_estimate(pcs[:, k], kernel="epanechnikov")
    lag_cors = xr.DataArray(
        lag_cors,
        dims=["lag", "component"],
        coords={
            "lag": np.arange(-max_lag, max_lag + 1),
            "component": np.arange(1, pcs.shape[1] + 1),
        },
    )
    kernel_densities = xr.DataArray(
        kernel_densities,
        dims=["point", "component", "arr_type"],
        coords={
            "component": np.arange(1, pcs.shape[1] + 1),
            "type": ["x", "density"],
        },
    )
    base_stats = xr.DataArray(
        [
            np.mean(pcs, axis=0),
            np.std(pcs, axis=0, ddof=1),
            sts.skew(pcs, axis=0),
            sts.kurtosis(pcs, axis=0),
        ],
        dims=["stat", "component"],
        coords={
            "stat": ["mean", "std", "skew", "kurt"],
            "component": np.arange(1, pcs.shape[1] + 1),
        },
    )
    return lag_cors, kernel_densities, base_stats


class EmpiricalModel(DataField):
    """
    Class for empirical model.
    """

    _additional_copy_attributes = [
        "no_levels",
        "low_freq",
        "input_pcs",
        "input_eofs",
        "orig_data_xr",
        "model_options",
    ]

    def __init__(self, no_levels, data):
        """
        :param no_levels: number of levels in the empirical model
        :type no_levels: int
        :param data: spatio-temporal data
        :type data: xr.DataArray
        """
        # set attributes
        self.no_levels = no_levels
        self.low_freq = None
        self.input_pcs = None
        self.input_eofs = None
        self.model_options = {}
        assert isinstance(
            data, xr.DataArray
        ), f"Data has to be xr.DataArray, got {type(data)}"
        self.orig_data_xr = deepcopy(data)

        super().__init__(data=data)

    def get_model_option(self, opt):
        return self.model_options.get(opt, False)

    def from_datafield(cls, no_levels, datafield):
        """
        Init EmpiricalModel from other datafield.
        """
        assert isinstance(datafield, DataField)
        emp_model = cls(no_levels, datafield.data).__finalize__(datafield)

        return emp_model

    def __finalize__(self, other, add_steps=None):
        """
        Copy additional attributes.
        """
        for attr in self._additional_copy_attributes:
            if hasattr(other, attr):
                setattr(self, attr, deepcopy(getattr(other, attr)))
        return super().__finalize__(other, add_steps=add_steps)

    @staticmethod
    def _pca(self, data_xr, pca_mean, n_comps):
        """
        Helper for PCA.

        :param data_xr: input data as DataArray
        :type data_xr: xr.DataArray
        :param pca_mean: temporal mean of the data
        :type pca_mean: xr.DataArray
        :param n_comps: number of components to extract
        :type n_comps: int|float|None
        :return: PCA class, principal components, orthogonal functions,
            explained variance
        :rtype: `sklearn.decomposition.PCA`, xr.DataArray, xr.DataArray,
            np.ndarray
        """
        pca_class = PCA(n_components=n_comps or 0.99)
        pcs = pca_class.fit_transform((data_xr - pca_mean).values)
        pcs = xr.DataArray(
            data=pcs.copy(),
            dims=["time", "component"],
            coords={
                "time": data_xr.time,
                "component": np.arange(1, pcs.shape[1] + 1),
            },
        )
        eofs = xr.DataArray(
            data=pca_class.components_.copy(),
            dims=["component", "space"],
            coords={
                "component": np.arange(1, pcs.shape[1] + 1),
                "space": data_xr.coords["space"],
            },
        ).unstack()
        eofs_full = np.empty((eofs.shape[0],) + self.data.shape[1:])
        eofs_full[:] = np.nan
        _, _, idx_lats = np.intersect1d(
            eofs.lats.values, self.lats, return_indices=True
        )
        _, _, idx_lons = np.intersect1d(
            eofs.lons.values, self.lons, return_indices=True
        )
        eofs_full[:, idx_lats.reshape((-1, 1)), idx_lons] = eofs.values
        eofs = xr.DataArray(
            data=eofs_full,
            dims=["component"] + self.dims_not_time,
            coords={
                "component": np.arange(1, pcs.shape[1] + 1),
                **self.coords_not_time,
            },
        )

        return pca_class, pcs, eofs, pca_class.explained_variance_ratio_.copy()

    def remove_low_freq_variability(
        self, mean_over, cos_weights=True, no_comps=None
    ):
        """
        Remove low-frequency variability (usually magnitude of decades) and
        store the signal in EOFs.

        :param mean_over: how many years to run over
        :type mean_over: float
        :param cos_weights: whether to use cosine reweighting
        :type cos_weights: bool
        :param no_comps: number of components for storing low-freq variability,
            if None will select number such that it keeps 99% of the variance
        """

        logging.info("Removing low frequency variability...")

        # rolling mean
        logging.debug("...running rolling mean over %d years..." % mean_over)
        smoothed = self.data.rolling(
            int(mean_over * (1.0 / self.dt("years"))), center=True
        ).reduce(np.nanmean)

        # cos-weighting
        if cos_weights:
            logging.debug("...cos-weighting...")
            smoothed *= self.cos_weights

        # pca on low-freq field
        logging.debug("...storing low-freq variability in PCA...")
        smoothed_flat_data = smoothed.stack(space=self.dims_not_time).dropna(
            dim="space", how="any"
        )
        low_freq_pca_mean = smoothed_flat_data.mean(dim="time")
        low_freq_pca, low_freq_pcs, low_freq_eofs, low_freq_var = self._pca(
            smoothed_flat_data, low_freq_pca_mean, no_comps
        )
        # save low freq
        self.low_freq = [
            low_freq_pca,
            low_freq_pcs,
            low_freq_eofs,
            low_freq_var,
            low_freq_pca_mean,
        ]
        logging.debug(
            f"...low-freq PCA retained {low_freq_var.sum():.2%} of original"
            " variance..."
        )

        # subtract from data
        logging.debug("...subtracting low-frequency variability from data...")
        # remove low-freq from data
        self.data -= smoothed
        with_cos_weights = " scaled by cos-weights " if cos_weights else ""
        self.process_steps += [
            f"removed low-freq variability over {mean_over} years"
            f"{with_cos_weights}, stored in {len(low_freq_pca['component'])}"
        ]

    def prepare_input(
        self, no_input_ts=20, anomalise=True, cos_weights=True, sel=None
    ):
        """
        Prepare input time series to model as PCs.

        :param no_input_ts: number of input PC timeseries
        :type no_input_ts: int
        :param anomalise: whether to anomalise data
        :type anomalise: bool
        :param cos_weights: whether to use cosine reweighting
        :type cos_weights: bool
        :param sel: whether to select specific PCs, if None, use all, uses
            numbering from 1
        :type sel: list|tuple|None
        """

        logging.info("Preparing input to the model as principal components...")

        no_input_ts = no_input_ts if sel is None else len(sel)
        self.model_options["input_anomalise"] = anomalise
        add_steps = []

        if anomalise:
            logging.debug("...anomalising...")
            self.climatological_mean = self.anomalise(inplace=True)
            add_steps += ["anomalise"]

        if cos_weights:
            logging.debug("...cos-weighting...")
            self.data *= self.cos_weights
            add_steps += ["cos-weighting"]

        logging.debug("...computing PCA...")
        pcs, eofs, var = self.pca(
            n_comps=no_input_ts if sel is None else sel[-1] + 1,
            return_nans=True,
        )
        if sel is not None:
            logging.debug(f"...selecting {', '.join(sel)} components...")
            pcs = pcs.sel({"component": sel})
            eofs = eofs.sel({"component": sel})
            var = [va for i, va in enumerate(var) if i + 1 in sel]
        logging.debug(f"...PCA retained {var.sum():.2%} variance...")
        add_steps += [f"input as {len(pcs['component'])} PCs"]

        self.input_eofs = eofs
        self.std_first_pc = np.std(pcs.isel({"component": 0}).values, ddof=1)
        # standardise PCs
        self.input_pcs = pcs / self.std_first_pc

        self.process_steps += add_steps

    # def combined_model(self, field):
    #     """
    #     Adds other field or prepared model to existing one, allowing to model multiple variables.
    #     Field is instance of EmpiricalModel with already created input_pcs.
    #     """

    #     try:
    #         shp = field.input_pcs.shape
    #         if self.verbose:
    #             print(
    #                 "Adding other %d input pcs of length %d to current one variable model..."
    #                 % (shp[0], shp[1])
    #             )
    #         if shp[1] != self.input_pcs.shape[1]:
    #             raise Exception(
    #                 "Combined model must have all variables of the same time series length!"
    #             )
    #     except:
    #         pass

    #     self.combined = True
    #     self.comb_std_pc1 = np.std(field.input_pcs[0, :], ddof=1)
    #     pcs_comb = field.input_pcs / self.comb_std_pc1

    #     self.copy_orig_input_pcs = self.input_pcs.copy()
    #     self.input_pcs = np.concatenate((self.input_pcs, pcs_comb), axis=0)
    #     if self.verbose:
    #         print(
    #             "... input pcs from other field added. Now we are training model on %d PCs..."
    #             % (self.input_pcs.shape[0])
    #         )

    @staticmethod
    def _get_xsin_xcos(len):
        """
        Helper function to get harmonics predictors with annual frequency.
        """
        logging.debug("...using harmonic predictors (with annual frequency)...")
        return np.sin(2 * np.pi * np.arange(len) / 12.0), np.cos(
            2 * np.pi * np.arange(len) / 12.0
        )

    @staticmethod
    def _build_quad_predictor(pcs):
        """
        Helper function to build quadratic predictor.
        """
        quad_pred = np.zeros(
            (
                pcs.shape[0],
                (pcs.shape[1] * (pcs.shape[1] - 1)) / 2,
            )
        )
        for t in range(pcs.shape[0]):
            q = np.tril(np.outer(pcs[t, :].T, pcs[t, :]), -1)
            quad_pred[t, :] = q[np.nonzero(q)]
        return quad_pred

    @staticmethod
    def _build_harmonic_predictor(x, xsin, xcos, quad_pred=None):
        """
        Helper function to build harmonic predictor.
        """
        if quad_pred:
            return np.c_[
                quad_pred,
                x,
                x * np.outer(xsin, np.ones(x.shape[1])),
                x * np.outer(xcos, np.ones(x.shape[1])),
                xsin,
                xcos,
            ]
        else:
            return np.c_[
                x,
                x * np.outer(xsin, np.ones(x.shape[1])),
                x * np.outer(xcos, np.ones(x.shape[1])),
                xsin,
                xcos,
            ]

    def train_model(
        self,
        harmonic_predictor="first",
        quadratic_model=False,
        delayed_model=False,
        method="partialLSQ",
        **kwargs,
    ):
        """
        Train the multi-level statistical model.

        :param harmonic_predictor: type of harmonic predictor to use, options:
            "first"
            "all"
            "none"
        :type harmonic_predictor: str
        :param quadratic_model: whether to train quadratic model or linear
        :type quadratic_model: bool
        :param delayed_model: whether linear part of the model should be
            considered as DDE with sigmoid type of response
        :type delayed_model: bool
        :param method: method for regression, options:
            "partialLSQ" - for partial least squares
            "linear" - for basic linear regressor
            "ridge" - for ridge regressor
            "bayes_ridge" - for Bayesian ridge regressor
        :type method: str
        :kwargs: possible keyword arguments:
            delay - delay for delayed model, in months
            kappa - coefficient for sigmoidal response in delayed model
        """
        if harmonic_predictor not in ["first", "none", "all"]:
            raise Exception("Unknown keyword for harmonic predictor")

        self.model_options["harmonic_predictor"] = harmonic_predictor
        self.model_options["quadratic_model"] = quadratic_model
        self.model_options["delayed_model"] = delayed_model

        logging.info(
            f"Training {self.no_levels}-level model using {method} regressor..."
        )

        pcs = deepcopy(self.input_pcs.values)
        pcs = pcs.T  # time x dim
        if delayed_model:
            # shorten time series because of delay
            self.delayed_model = True
            self.delay = kwargs.get("delay", 8)
            self.kappa = kwargs.get("kappa", 50.0)
            if self.verbose:
                logging.debug(
                    f"...training delayed model on main level with delay "
                    f"{self.delay} months and kappa={self.kappa}..."
                )
            pcs_delay = pcs[: -self.delay, :].copy()
            pcs = pcs[self.delay :, :]

        if harmonic_predictor in ["all", "first"]:
            xsin, xcos = self._get_xsin_xcos(pcs.shape[0])

        residuals = {}
        fit_mat = {}

        for level in range(self.no_levels):
            logging.debug(f"...training {level+1}/{self.no_levels} levels...")

            # figure out matrix size
            # as extended vector + intercept
            fit_mat_size = pcs.shape[1] * (level + 1) + 1
            if level == 0:
                if harmonic_predictor in ["first", "all"]:
                    fit_mat_size += 2 * pcs.shape[1] + 2
                if quadratic_model and level == 0:
                    fit_mat_size += (pcs.shape[1] * (pcs.shape[1] - 1)) / 2
            elif level > 0:
                if harmonic_predictor == "all":
                    fit_mat_size += (level + 1) * 2 * pcs.shape[1] + 2

            # response variables - y (dx/dt)
            logging.debug("...preparing response variables...")
            y = np.zeros_like(pcs)
            if level == 0:
                y[:-1, :] = np.diff(pcs, axis=0)
            else:
                y[:-1, :] = np.diff(residuals[level - 1], axis=0)
            y[-1, :] = y[-2, :]

            fit_mat[level] = np.zeros((fit_mat_size, pcs.shape[1]))
            residuals[level] = np.zeros_like(pcs)

            for k in range(pcs.shape[1]):
                # prepare predictor
                x = deepcopy(pcs)
                for lev in range(level):
                    x = np.c_[x, residuals[lev]]
                if level == 0:
                    if quadratic_model:
                        quad_pred = self._build_quad_predictor(pcs)
                    if delayed_model:
                        x = np.tanh(self.kappa * pcs_delay)
                    if harmonic_predictor in ["all", "first"]:
                        if quadratic_model:
                            x = self._build_harmonic_predictor(
                                x, xsin, xcos, quad_pred
                            )
                        else:
                            x = self._build_harmonic_predictor(
                                x, xsin, xcos, None
                            )
                    else:
                        if quadratic_model:
                            x = np.c_[quad_pred, x]
                else:
                    if harmonic_predictor == "all":
                        x = self._build_harmonic_predictor(x, xsin, xcos, None)

                # regularize and regress
                if method == "partialLSQ":
                    x -= np.mean(x, axis=0)
                    ux, sx, vx = np.linalg.svd(x, False)
                    optimal = min(ux.shape[1], 25)
                    b_aux, residuals[level][:, k] = _partial_least_squares(
                        x, y[:, k], ux, sx, vx.T, optimal, True
                    )
                else:
                    if method == "bayes_ridge":
                        self.regressor = BayesianRidge(fit_intercept=True)
                    elif method == "linear":
                        self.regressor = LinearRegression(fit_intercept=True)
                    elif method == "ridge":
                        self.regressor = Ridge(fit_intercept=True, alpha=0.5)
                    else:
                        raise Exception("Unknown regressing method")

                    self.regressor.fit(x, y[:, k])
                    b_aux = np.append(
                        self.regressor.coef_, self.regressor.intercept_
                    )
                    residuals[level][:, k] = y[:, k] - self.regressor.predict(x)

                # store results
                fit_mat[level][:, k] = b_aux

                if (k + 1) % 10 == 0:
                    logging.debug(
                        f"...{k+1}/{pcs.shape[1]} finished fitting..."
                    )

            # check for negative definiteness
            negdef = {}
            for lev, e, pos in zip(
                fit_mat.keys()[::-1],
                range(len(fit_mat.keys())),
                range(len(fit_mat.keys()) - 1, -1, -1),
            ):
                negdef[lev] = fit_mat[lev][
                    pos * pcs.shape[1] : (pos + 1) * pcs.shape[1]
                ]
                for a in range(pos - 1, -1, -1):
                    negdef[lev] = np.c_[
                        negdef[lev],
                        fit_mat[lev][a * pcs.shape[1] : (a + 1) * pcs.shape[1]],
                    ]
                for a in range(e):
                    negdef[lev] = np.c_[negdef[lev], np.eye(pcs.shape[1])]
            grand_negdef = np.concatenate(
                [negdef[a] for a in negdef.keys()], axis=0
            )
            d, _ = np.linalg.eig(grand_negdef)
            logging.debug(f"...maximum eigenvalue: {max(np.real(d)):.4f}")

        self.residuals = residuals
        self.fit_mat = fit_mat

    @staticmethod
    def _get_spatial_cov_white_noise(resid):
        """
        Helper method to get spatial covariance of white noise from residuals.
        """
        Q = np.cov(resid, rowvar=0)
        return np.linalg.cholesky(Q).T

    def _get_spatial_cov_seasonal_noise(self, resid, n_harmonics, n_pcs):
        """
        Helper method to get spatial covariance of seasonal noise from
        residuals.
        """
        if self.get_model_option("delayed_model"):
            resid_delayed = resid[-(resid.shape[0] // 12) * 12 :].copy()
            rr_last = np.reshape(
                resid_delayed,
                (
                    12,
                    resid.shape[0] // 12,
                    resid.shape[1],
                ),
                order="F",
            )
        else:
            rr_last = np.reshape(
                resid,
                (
                    12,
                    resid.shape[0] // 12,
                    resid.shape[1],
                ),
                order="F",
            )
        rr_last_std = np.nanstd(rr_last, axis=1, ddof=1)
        predictors = np.zeros((12, 2 * n_harmonics + 1))
        for nh in range(n_harmonics):
            predictors[:, 2 * nh] = np.cos(
                2 * np.pi * (nh + 1) * np.arange(12) / 12
            )
            predictors[:, 2 * nh + 1] = np.sin(
                2 * np.pi * (nh + 1) * np.arange(12) / 12
            )
        predictors[:, -1] = np.ones((12,))
        bamp = np.zeros((predictors.shape[1], n_pcs))
        for k in range(bamp.shape[1]):
            bamp[:, k] = np.linalg.lstsq(predictors, rr_last_std[:, k])[0]
        rr_last_std_ts = np.dot(predictors, bamp)
        rr_last_std_ts = np.repeat(
            rr_last_std_ts,
            repeats=resid.shape[0] // 12,
            axis=0,
        )
        if self.get_model_option("delayed_model"):
            resid_delayed /= rr_last_std_ts
            Q = np.cov(resid_delayed, rowvar=0)
        else:
            resid /= rr_last_std_ts
            Q = np.cov(resid, rowvar=0)

        return np.linalg.cholesky(Q).T

    def integrate_model(
        self,
        n_realizations,
        integration_length=None,
        noise_type="white",
        sigma=1.0,
        n_workers=cpu_count(),
        diagnostics=True,
    ):
        """
        Integrate trained model.

        :param n_realizations: number of realizations to integrate
        :type n_realizations: int
        :param integration_length: integration length, if None, will use the
            same length as original data
        :type integration_length: int
        :param noise_type: noise type to use for integration, options:
            "white" - classic white noise, spatial correlation by cov. matrix of
                thelast level of residuals
            "cond" - find n_samples closest to the current space in subset of
                n_pcs and use their cov. matrix
            "seasonal" - seasonal dependence of the residuals, fit n_harm
                harmonics of annual cycle, could also be used with cond
            "cond" and "seasonal" can be combined
        :type noise_type: str
        :param sigma: noise variance
        :type sigma: float
        :param n_workers: number of workers for parallel integration
        :type n_workers: int
        :param diagnostics: whether to save also statistics of all runs
        :type diagnostics: bool
        """
        logging.info("Preparing to integrate trained model...")

        pcs = deepcopy(self.input_pcs.values)
        pcs = pcs.T  # time x dim

        pcmax = np.amax(pcs, axis=0)
        pcmin = np.amin(pcs, axis=0)
        self.varpc = np.var(pcs, axis=0, ddof=1)

        integration_length = integration_length or pcs.shape[0]
        # self.diagnostics = diagnostics

        if self.get_model_option("harmonic_predictor") in ["all", "first"]:
            xsin, xcox = self._get_xsin_xcos(integration_length)

        logging.debug("...preparing noise forcing...")

        # self.sigma = sigma
        if isinstance(noise_type, str):
            if noise_type not in ["white", "cond", "seasonal"]:
                raise Exception(
                    "Unknown noise type to be used as forcing. Use 'white', "
                    "'cond', or 'seasonal'."
                )
        elif isinstance(noise_type, list):
            noise_type = frozenset(noise_type)
            if not noise_type.issubset(set(["white", "cond", "seasonal"])):
                raise Exception(
                    "Unknown noise type to be used as forcing. Use 'white', "
                    "'cond', or 'seasonal'."
                )

        last_level_res = self.residuals[max(self.residuals.keys())]
        if noise_type == "white":
            logging.debug("...using spatially correlated white noise...")
            rr = self._get_spatial_cov_white_noise(last_level_res)

        if "seasonal" in noise_type:
            n_harmonics = 5
            logging.debug(
                f"...fitting {n_harmonics} harmonics to estimate seasonal "
                "modulation of last level's residual..."
            )
            rr = self._get_spatial_cov_seasonal_noise(
                last_level_res, n_harmonics=n_harmonics, n_pcs=pcs.shape[1]
            )

        # if diagnostics:
        #     logging.debug("...running diagnostics for the data...")

        #     # init for integrations
        #     lag_cors_int = np.zeros([n_realizations] + list(lag_cors.shape))
        #     kernel_densities_int = np.zeros(
        #         [n_realizations] + list(kernel_densities.shape)
        #     )
        #     stat_moments_int = np.zeros(
        #         (4, n_realizations, pcs.shape[1])
        #     )  # mean, variance, skewness, kurtosis
        #     int_corr_scale_int = np.zeros((n_realizations, pcs.shape[1]))

        diagpc = np.diag(np.std(pcs, axis=0, ddof=1))
        maxpc = np.amax(np.abs(pcs))
        diagres = {}
        maxres = {}
        for lev in self.residuals.keys():
            diagres[lev] = np.diag(np.std(self.residuals[lev], axis=0, ddof=1))
            maxres[lev] = np.amax(np.abs(self.residuals[lev]))

        # self.pcs = pcs
        logging.debug(
            f"...running integration of {n_realizations} realizations using "
            f"{n_workers} workers..."
        )
        pool = ProcessingPool(n_workers)

        precomputed_noise = []
        for _ in range(n_realizations):
            r = {}
            for lev in self.fit_mat.keys():
                if lev == 0:
                    if self.delayed_model:
                        r[lev] = np.dot(
                            diagpc,
                            np.random.normal(
                                0, sigma, (pcs.shape[1], self.delay)
                            ),
                        )
                    else:
                        r[lev] = np.dot(
                            np.random.normal(0, sigma, (pcs.shape[1],)),
                            diagpc,
                        )
                else:
                    if self.delayed_model:
                        r[lev] = np.dot(
                            diagres[lev - 1],
                            np.random.normal(
                                0, sigma, (pcs.shape[1], self.delay)
                            ),
                        )
                    else:
                        r[lev] = np.dot(
                            np.random.normal(0, sigma, (pcs.shape[1],)),
                            diagres[lev - 1],
                        )
            precomputed_noise.append(r)
        args = [
            [i, rnd] for i, rnd in zip(range(n_realizations), precomputed_noise)
        ]
        results = list(pool.imap(self._process_integration, args))

        del args
        pool.close()
        pool.join()

        self.integration_results = np.zeros(
            (n_realizations, pcs.shape[1], integration_length)
        )
        num_exploding = np.zeros((n_realizations,))

        # if self.diagnostics:
        #     # x, num_exploding, xm, xv, xs, xk, lc, kden, ict
        #     for i, x, num_expl, xm, xv, xs, xk, lc, kden, ict in results:
        #         self.integration_results[i, ...] = x.T
        #         num_exploding[i] = num_expl
        #         stat_moments_int[0, i, :] = xm
        #         stat_moments_int[1, i, :] = xv
        #         stat_moments_int[2, i, :] = xs
        #         stat_moments_int[3, i, :] = xk
        #         lag_cors_int[i, ...] = lc
        #         kernel_densities_int[i, ...] = kden
        #         int_corr_scale_int[i, ...] = ict
        # else:
        #     for i, x, num_expl in results:
        #         self.integration_results[i, ...] = x.T
        #         self.num_exploding[i] = num_expl

        logging.info("All done.")
        logging.info(
            f"There was {np.sum(self.num_exploding)} expolding integration "
            f"chunks in {n_realizations} realizations."
        )

    def _process_integration(self, args):
        """
        Helper for parallel integration.
        """
        i, rnd = args
        num_exploding = 0
        repeats = 20
        xx = {}
        x = {}
        for lev in self.fit_mat.keys():
            xx[lev] = np.zeros((repeats, self.input_pcs.shape[0]))
            if self.delayed_model:
                xx[lev][: self.delay, :] = rnd[lev].T
            else:
                xx[lev][0, :] = rnd[lev]

            x[lev] = np.zeros((self.int_length, self.input_pcs.shape[0]))
            if self.delay_model:
                x[lev][: self.delay, :] = xx[lev][: self.delay, :]
            else:
                x[lev][0, :] = xx[lev][0, :]

        step0 = 0
        step = self.delay if self.delayed_model else 1
        blow_counter = 0
        zz = {}

        for n in range(repeats * int(np.ceil(self.int_length / repeats))):
            for k in range(1, repeats):
                if (self.delay_model and k < self.delay) and step == self.delay:
                    continue
                if blow_counter >= 10:
                    raise Exception("Model blowed up 10 times.")
                if step >= self.int_length:
                    break
                # prepare predictors
                for lev in self.fit_mat.keys():
                    zz[lev] = xx[0][k - 1, :]
                    for lr in range(lev):
                        zz[lev] = np.r_[zz[lev], xx[lr + 1][k - 1, :]]
                for lev in self.fit_mat.keys():
                    if lev == 0:
                        if self.quad:
                            q = np.tril(np.outer(zz[lev].T, zz[lev]), -1)
                            quad_pred = q[np.nonzero(q)]
                        if self.delay_model:
                            zz[lev] = np.tanh(
                                self.kappa * x[lev][step - self.delay, :]
                            )
                        if self.harmonic_pred in ["all", "first"]:
                            if self.quad:
                                zz[lev] = np.r_[
                                    quad_pred,
                                    zz[lev],
                                    zz[lev] * self.xsin[step],
                                    zz[lev] * self.xcos[step],
                                    self.xsin[step],
                                    self.xcos[step],
                                    1,
                                ]
                            else:
                                zz[lev] = np.r_[
                                    zz[lev],
                                    zz[lev] * self.xsin[step],
                                    zz[lev] * self.xcos[step],
                                    self.xsin[step],
                                    self.xcos[step],
                                    1,
                                ]
                        else:
                            if self.quad:
                                zz[lev] = np.r_[quad_pred, zz[lev], 1]
                            else:
                                zz[lev] = np.r_[zz[lev], 1]
                    else:
                        if self.harmonic_pred == "all":
                            zz[lev] = np.r_[
                                zz[lev],
                                zz[lev] * self.xsin[step],
                                zz[lev] * self.xcos[step],
                                self.xsin[step],
                                self.xcos[step],
                                1,
                            ]
                        else:
                            zz[lev] = np.r_[zz[lev], 1]

                if "cond" in self.noise_type:
                    n_PCs = 1
                    n_samples = 100
                    if not self.combined:
                        ndx = np.argsort(
                            np.sum(
                                np.power(
                                    self.pcs[:, :n_PCs] - xx[0][k - 1, :n_PCs],
                                    2,
                                ),
                                axis=1,
                            )
                        )
                        Q = np.cov(
                            self.last_level_res[ndx[:n_samples], :], rowvar=0
                        )
                        rr = np.linalg.cholesky(Q).T
                    elif self.combined:
                        ndx1 = np.argsort(
                            np.sum(
                                np.power(
                                    self.pcs[:, :n_PCs] - xx[0][k - 1, :n_PCs],
                                    2,
                                ),
                                axis=1,
                            )
                        )
                        ndx2 = np.argsort(
                            np.sum(
                                np.power(
                                    self.pcs[
                                        :,
                                        self.no_input_ts : self.no_input_ts
                                        + n_PCs,
                                    ]
                                    - xx[0][
                                        k - 1,
                                        self.no_input_ts : self.no_input_ts
                                        + n_PCs,
                                    ],
                                    2,
                                ),
                                axis=1,
                            )
                        )
                        res1 = self.last_level_res[ndx1[:n_samples], :]
                        res2 = self.last_level_res[ndx2[:n_samples], :]
                        Q = np.cov(
                            np.concatenate((res1, res2), axis=0), rowvar=0
                        )
                        rr = np.linalg.cholesky(Q).T

                # integration step
                for lev in sorted(self.fit_mat, reverse=True):
                    if lev == self.no_levels - 1:
                        forcing = np.dot(
                            rr,
                            np.random.normal(0, self.sigma, (rr.shape[0],)).T,
                        )
                        if "seasonal" in self.noise_type:
                            forcing *= self.rr_last_std_ts[
                                step % self.rr_last_std_ts.shape[0], :
                            ]
                    else:
                        forcing = xx[lev + 1][k, :]
                    xx[lev][k, :] = (
                        xx[lev][k - 1, :]
                        + np.dot(zz[lev], self.fit_mat[lev])
                        + forcing
                    )

                step += 1

            # check if integration blows
            if np.amax(np.abs(xx[0])) <= 2 * self.maxpc and not np.any(
                np.isnan(xx[0])
            ):
                for lev in self.fit_mat.keys():
                    x[lev][step - repeats + 1 : step, :] = xx[lev][1:, :]
                    # set first to last
                    xx[lev][0, :] = xx[lev][-1, :]
            else:
                for lev in self.fit_mat.keys():
                    if lev == 0:
                        xx[lev][0, :] = np.dot(
                            np.random.normal(
                                0, self.sigma, (self.input_pcs.shape[0],)
                            ),
                            self.diagpc,
                        )
                    else:
                        xx[lev][0, :] = np.dot(
                            np.random.normal(
                                0, self.sigma, (self.input_pcs.shape[0],)
                            ),
                            self.diagres[lev - 1],
                        )
                if step != step0:
                    num_exploding += 1
                    step0 = step
                step -= repeats + 1
                blow_counter += 1

        x = x[0].copy()

        # center
        x -= np.mean(x, axis=0)

        # preserve total energy level
        x *= np.sqrt(np.sum(self.varpc) / np.sum(np.var(x, axis=0, ddof=1)))

        if self.diagnostics:
            xm = np.mean(x, axis=0)
            xv = np.var(x, axis=0, ddof=1)
            xs = sts.skew(x, axis=0)
            xk = sts.kurtosis(x, axis=0)

            lc = np.zeros((2 * self.max_lag + 1, self.input_pcs.shape[0]))
            kden = np.zeros((100, self.input_pcs.shape[0], 2))
            for k in range(self.input_pcs.shape[0]):
                lc[:, k] = cross_correlation(
                    x[:, k], x[:, k], max_lag=self.max_lag
                )
                kden[:, k, 0], kden[:, k, 1] = kdensity_estimate(
                    x[:, k], kernel="epanechnikov"
                )
            ict = np.sum(np.abs(lc), axis=0)

            return i, x, num_exploding, xm, xv, xs, xk, lc, kden, ict

        else:
            return i, x, num_exploding

    def reconstruct_simulated_field(
        self, lats=None, lons=None, mean=False, save_by_one=None
    ):
        """
        Reconstructs 3D geofield from simulated PCs.
        If lats and/or lons is given, will save only a cut from 3D data.
        If mean is True, will save only one time series as spatial mean (e.g. indices).
        If save_by_one is not None but a string, saves a file for each reconstruction.
        """

        self.reconstructions = []

        if self.verbose:
            print("reconstructing 3D geo fields...")
            if save_by_one is not None:
                if not isinstance(save_by_one, str):
                    raise Exception(
                        "save_by_one must be a string! It is a filename base for files."
                    )

                print(
                    "...individual reconstructed fields will be saved in separate files..."
                )

        if self.combined:
            combined_pcs = self.integration_results[
                :, self.no_input_ts :, :
            ].copy()
            # "destandardise" combined PCs
            self.reconstruction_comb_pcs = combined_pcs * self.comb_std_pc1
            # cut only "our" PCs
            self.integration_results = self.integration_results[
                :, : self.no_input_ts, :
            ]

        if (lats is not None) or (lons is not None):
            lat_ndx, lon_ndx = self.select_lat_lon(
                lats, lons, apply_to_data=False
            )

        for n in range(self.integration_results.shape[0]):
            if self.verbose and (n + 1) % 5 == 0:
                print(
                    "...processing field %d/%d..."
                    % (n + 1, self.integration_results.shape[0])
                )
            # "destandardise" PCs
            pcs = self.integration_results[n, ...] * self.std_first_pc
            # invert PCA analysis with modelled PCs
            reconstruction = self.invert_pca(
                self.input_eofs, pcs
            )  # time x lats x lons
            # if anomalised, return seasonal climatology
            if self.input_anom:
                # works only with monthly data
                for mon in range(12):
                    reconstruction[mon::12, ...] += self.clim_mean[mon, ...]
            # add low freq variability if removed
            if self.low_freq is not None:
                # get the low freq field already with pca mean added
                low_freq_field = self.invert_pca(
                    self.low_freq[0],
                    self.low_freq[1],
                    pca_mean=self.low_freq[3],
                )
                if low_freq_field.shape[0] >= reconstruction.shape[0]:
                    # if integration - modelled field - is shorter than or equal to low_freq_field length
                    reconstruction += low_freq_field[
                        : reconstruction.shape[0], ...
                    ]
                else:
                    # if integration is longer
                    factor = reconstruction.shape[0] // low_freq_field.shape[0]
                    low_freq_field = np.repeat(
                        low_freq_field, repeats=factor, axis=0
                    )
                    low_freq_field = np.append(
                        low_freq_field,
                        low_freq_field[
                            : reconstruction.shape[0] - low_freq_field.shape[0]
                        ],
                        axis=0,
                    )
                    reconstruction += low_freq_field

            if (lats is not None) or (lons is not None):
                reconstruction = reconstruction[:, lat_ndx, :]
                reconstruction = reconstruction[:, :, lon_ndx]
                if mean:
                    reconstruction = np.nanmean(reconstruction, axis=(1, 2))

            if save_by_one is not None:
                if self.verbose:
                    print("...saving file...")

                to_save = {
                    "reconstruction": reconstruction,
                    "lats": self.lats,
                    "lons": self.lons,
                }
                # comment/uncomment your preference
                ## mat file
                import scipy.io as sio

                sio.savemat("%s-%d.mat" % (save_by_one, n), to_save)

                ## cPickle binary file
                # import cPickle
                # with open("%s-%d.mat" % (save_by_one, n), "wb") as f:
                #     cPickle.dump(to_save, f, protocol = cPickle.HIGHEST_PROTOCOL)
            else:
                self.reconstructions.append(reconstruction)
        if save_by_one is None:
            self.reconstructions = np.array(self.reconstructions)

        if self.verbose:
            print("...reconstruction done.")

    def save(self, fname, save_all=False, mat=False):
        """
        Saves the field or just the result to file.
        if save_all is False, saves just the result (self.reconstructions) with lats and lons, if True, saves whole class
        if mat is True, saves to matlab ".mat" file
        """

        if (
            isinstance(self.reconstructions, list)
            and len(self.reconstructions) == 0
        ):
            raise Exception(
                "List of reconstructions is empty! You've probably chosen to save individual reconstructions in separate files!"
            )

        if self.verbose:
            print("saving to file: %s" % fname)

        if save_all:
            to_save = self.__dict__
            if mat:
                # scipy.io cannot save None type to mat file
                to_save = {}
                for i in self.__dict__:
                    if self.__dict__[i] is not None:
                        to_save[i] = self.__dict__[i]
        else:
            to_save = {
                "reconstructions": self.reconstructions,
                "lats": self.lats,
                "lons": self.lons,
            }

        if not mat:
            # save to bin - cPickle
            import cPickle

            if fname[-4:] == ".bin":
                fname += ".bin"

            with open(fname, "wb") as f:
                cPickle.dump(to_save, f, protocol=cPickle.HIGHEST_PROTOCOL)

        else:
            # save to mat
            import scipy.io as sio

            sio.savemat(fname, mdict=to_save, appendmat=True)

        if self.verbose:
            print("saved.")
