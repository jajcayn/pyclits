"""
created on Aug 21, 2016

@author: Nikola Jajcay, jajcay(at)cs.cas.cz

last update on Sep 22, 2017

Based on Kondrashov D., Kravtsov S., Robertson A. W. and Ghil M., J. Climate, 18, 2005.
"""


import numpy as np
import scipy.stats as sts
from functions import cross_correlation, kdensity_estimate
from geofield import DataField


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


class EmpiricalModel(DataField):
    """
    Class holds the geo data and is able to fit / train and integrate statistical model
    as in Kravtsov et al., J. Climate, 18, 4404 - 4424 (2005).
    Working with monthly data.
    """

    def __init__(self, no_levels, verbose=False):
        """
        Init function.
        """

        DataField.__init__(self)
        self.no_levels = no_levels
        self.low_freq = None
        self.input_pcs = None
        self.input_eofs = None
        self.verbose = verbose
        self.combined = False
        self.delay_model = False

    def load_geo_data(
        self,
        fname,
        varname,
        start_date,
        end_date,
        lats=None,
        lons=None,
        dataset="NCEP",
        anom=False,
    ):
        """
        Loads geo data, makes sure the data is monthly.
        """

        self.load(fname, varname, dataset=dataset, print_prog=False)
        self.select_date(start_date, end_date)
        self.select_lat_lon(lats, lons)
        if anom:
            self.anomalise()
        if np.abs(self.time[1] - self.time[0]) <= 1:
            raise Exception("Model works only with monthly data.")
        if self.verbose:
            print(
                "Data loaded with shape %s and time span %s -- %s."
                % (
                    self.data.shape,
                    self.get_date_from_ndx(0),
                    self.get_date_from_ndx(-1),
                )
            )

    def copy_existing_datafield(self, g):
        """
        Copies existing DataField instance to this model.
        """

        self.data = g.data.copy()
        self.time = g.time.copy()
        if np.abs(self.time[1] - self.time[0]) <= 1:
            raise Exception("Model now works only with monthly data.")
        self.lats = g.lats.copy()
        self.lons = g.lons.copy()
        self.nans = g.nans
        if self.verbose:
            print(
                "DataField copied to model. Shape of the data is %s, time span is %s -- %s including."
                % (
                    self.data.shape,
                    self.get_date_from_ndx(0),
                    self.get_date_from_ndx(-1),
                )
            )

    def remove_low_freq_variability(
        self, mean_over, cos_weights=True, no_comps=None
    ):
        """
        Removes low-frequency variability (usually magnitude of decades) and
        stores the signal in EOFs.
        mean_over in years, cos_weights whether to use cosine reweighting
        if no_comps is None, keeps number such 99% of variance is described.
        """

        if self.verbose:
            print("removing low frequency variability...")

        window = int((mean_over / 2.0) * 12.0)

        # boxcar mean
        smoothed = np.zeros_like(self.data)
        if self.verbose:
            print("...running boxcar mean over %d years..." % mean_over)
        for t in range(self.time.shape[0]):
            smoothed[t, ...] = np.nanmean(
                self.data[
                    max(t - window, 0) : min(t + window, self.time.shape[0]),
                    ...,
                ],
                axis=0,
            )

        # cos-weighting
        if cos_weights:
            if self.verbose:
                print("...scaling by square root of cosine of latitude...")
            cos_w = self.latitude_cos_weights()
            smoothed *= cos_w

        # pca on low-freq field
        if no_comps is not None:
            if self.verbose:
                print(
                    "...storing low frequency variability in %d EOFs..."
                    % no_comps
                )
            eofs, pcs, var, pca_mean = self.pca_components(
                n_comps=no_comps, field=smoothed
            )
            if self.verbose:
                print(
                    "...which explain %.2f%% of total low frequency variability..."
                    % (np.sum(var) * 100.0)
                )
        elif no_comps is None:
            if self.verbose:
                print(
                    "...storing low frequency variability in EOFs such that they explain 99% of variability..."
                )
            eofs, pcs, var, pca_mean = self.pca_components(
                n_comps=20, field=smoothed
            )
            idx = np.where(np.cumsum(var) > 0.99)[0][0] + 1
            eofs, pcs, var = eofs[:idx, ...], pcs[:idx, ...], var[:idx]
        self.low_freq = [eofs, pcs, var, pca_mean]

        # subtract from data
        if self.verbose:
            print("...subtracting from data...")
        if self.nans:
            self.data = self.filter_out_NaNs()[0]
            self.data -= pca_mean
            self.data = self.return_NaNs_to_data(self.data)
        else:
            self.flatten_field()
            self.data -= pca_mean
            self.reshape_flat_field()
        temp = self.flatten_field(eofs)
        self.flatten_field()
        self.data -= np.dot(temp.T, pcs).T
        self.reshape_flat_field()
        if self.verbose:
            print("done.")

    def prepare_input(
        self, anom=True, no_input_ts=20, cos_weights=True, sel=None
    ):
        """
        Prepares input time series to model as PCs.
        if sel is not None, selects those PCs as input (sel pythonic, starting with 0).
        """

        if self.verbose:
            print("preparing input to the model...")

        self.no_input_ts = no_input_ts if sel is None else len(sel)
        self.input_anom = anom

        if anom:
            if self.verbose:
                print("...anomalising...")
            self.clim_mean = self.anomalise()

        if cos_weights:
            if self.verbose:
                print("...scaling by square root of cosine of latitude...")
            cos_w = self.latitude_cos_weights()
            self.data *= cos_w

        if sel is None:
            if self.verbose:
                print(
                    "...selecting %d first principal components as input time series..."
                    % (no_input_ts)
                )
            eofs, pcs, var = self.pca_components(no_input_ts)
            self.input_pcs = pcs
            self.input_eofs = eofs
            if self.verbose:
                print(
                    "...and they explain %.2f%% of variability..."
                    % (np.sum(var) * 100.0)
                )
        else:
            if self.verbose:
                print(
                    "...selecting %d principal components described in 'sel' variable..."
                    % (len(sel))
                )
            eofs, pcs, var = self.pca_components(sel[-1] + 1)
            self.input_pcs = pcs[sel, :]
            self.input_eofs = eofs[sel, ...]
            if self.verbose:
                print(
                    "...and they explain %.2f%% of variability..."
                    % (np.sum(var[sel]) * 100.0)
                )

        # standardise PCs
        self.std_first_pc = np.std(self.input_pcs[0, :], ddof=1)
        self.input_pcs /= self.std_first_pc

        if self.verbose:
            print("done.")

    def combined_model(self, field):
        """
        Adds other field or prepared model to existing one, allowing to model multiple variables.
        Field is instance of EmpiricalModel with already created input_pcs.
        """

        try:
            shp = field.input_pcs.shape
            if self.verbose:
                print(
                    "Adding other %d input pcs of length %d to current one variable model..."
                    % (shp[0], shp[1])
                )
            if shp[1] != self.input_pcs.shape[1]:
                raise Exception(
                    "Combined model must have all variables of the same time series length!"
                )
        except:
            pass

        self.combined = True
        self.comb_std_pc1 = np.std(field.input_pcs[0, :], ddof=1)
        pcs_comb = field.input_pcs / self.comb_std_pc1

        self.copy_orig_input_pcs = self.input_pcs.copy()
        self.input_pcs = np.concatenate((self.input_pcs, pcs_comb), axis=0)
        if self.verbose:
            print(
                "... input pcs from other field added. Now we are training model on %d PCs..."
                % (self.input_pcs.shape[0])
            )

    def train_model(
        self,
        harmonic_pred="first",
        quad=False,
        delay_model=False,
        regressor="partialLSQ",
    ):
        """
        Train the model.
        harmonic_pred could have values 'first', 'all', 'none'
        if quad, train quadratic model, else linear
        if delay_model, the linear part of the model will be considered DDE with sigmoid type of response,
        inspired by DDE model of ENSO (Ghil) - delayed feedback.
        regression will be one of
            'partialLSQ' - for partial least squares
            'linear' - for basic linear regressor [sklearn]
            'ridge' - fir ridge regressor [sklearn]
            'bayes_ridge' - for Bayesian ridge regressor [sklearn]
        """

        self.harmonic_pred = harmonic_pred
        self.quad = quad

        if self.verbose:
            print(
                "now training %d-level model using %s regressor..."
                % (self.no_levels, regressor)
            )

        pcs = self.input_pcs.copy()

        if harmonic_pred not in ["first", "none", "all"]:
            raise Exception(
                "Unknown keyword for harmonic predictor, please use: 'first', 'all' or 'none'."
            )

        if quad and self.verbose:
            print("...training quadratic model...")

        pcs = pcs.T  # time x dim
        if delay_model:
            # shorten time series because of delay
            self.delay_model = True
            self.delay = 8  # months
            self.kappa = 50.0
            if self.verbose:
                print(
                    "...training delayed model on main level with delay %d months and kappa %.3f..."
                    % (self.delay, self.kappa)
                )
            pcs_delay = pcs[: -self.delay, :].copy()
            pcs = pcs[self.delay :, :]

        if harmonic_pred in ["all", "first"]:
            if self.verbose:
                print("...using harmonic predictors (with annual frequency)...")
            xsin = np.sin(2 * np.pi * np.arange(pcs.shape[0]) / 12.0)
            xcos = np.cos(2 * np.pi * np.arange(pcs.shape[0]) / 12.0)

        residuals = {}
        fit_mat = {}

        for level in range(self.no_levels):

            if self.verbose:
                print(
                    "...training %d. out of %d levels..."
                    % (level + 1, self.no_levels)
                )

            fit_mat_size = (
                pcs.shape[1] * (level + 1) + 1
            )  # as extended vector + intercept
            if level == 0:
                if harmonic_pred in ["first", "all"]:
                    fit_mat_size += 2 * pcs.shape[1] + 2  # harm
                if quad and level == 0:
                    fit_mat_size += (
                        pcs.shape[1] * (pcs.shape[1] - 1)
                    ) / 2  # quad
            elif level > 0:
                if harmonic_pred == "all":
                    fit_mat_size += (level + 1) * 2 * pcs.shape[1] + 2

            # response variables -- y (dx/dt)
            if self.verbose:
                print("...preparing response variables...")
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
                x = pcs.copy()
                for l in range(level):
                    x = np.c_[x, residuals[l]]
                if level == 0:
                    if quad:
                        quad_pred = np.zeros(
                            (
                                pcs.shape[0],
                                (pcs.shape[1] * (pcs.shape[1] - 1)) / 2,
                            )
                        )
                        for t in range(pcs.shape[0]):
                            q = np.tril(np.outer(pcs[t, :].T, pcs[t, :]), -1)
                            quad_pred[t, :] = q[np.nonzero(q)]
                    if self.delay_model:
                        x = np.tanh(self.kappa * pcs_delay)
                    if harmonic_pred in ["all", "first"]:
                        if quad:
                            x = np.c_[
                                quad_pred,
                                x,
                                x * np.outer(xsin, np.ones(x.shape[1])),
                                x * np.outer(xcos, np.ones(x.shape[1])),
                                xsin,
                                xcos,
                            ]
                        else:
                            x = np.c_[
                                x,
                                x * np.outer(xsin, np.ones(x.shape[1])),
                                x * np.outer(xcos, np.ones(x.shape[1])),
                                xsin,
                                xcos,
                            ]
                    else:
                        if quad:
                            x = np.c_[quad_pred, x]
                else:
                    if harmonic_pred == "all":
                        x = np.c_[
                            x,
                            x * np.outer(xsin, np.ones(x.shape[1])),
                            x * np.outer(xcos, np.ones(x.shape[1])),
                            xsin,
                            xcos,
                        ]

                # regularize and regress
                if regressor != "partialLSQ":
                    from sklearn import linear_model as lm

                    if regressor == "bayes_ridge":
                        self.regressor = lm.BayesianRidge(fit_intercept=True)
                    elif regressor == "linear":
                        self.regressor = lm.LinearRegression(fit_intercept=True)
                    elif regressor == "ridge":
                        self.regressor = lm.Ridge(fit_intercept=True, alpha=0.5)
                    else:
                        raise Exception(
                            "Unknown regressor, please check documentation!"
                        )

                    self.regressor.fit(x, y[:, k])
                    b_aux = np.append(
                        self.regressor.coef_, self.regressor.intercept_
                    )
                    residuals[level][:, k] = y[:, k] - self.regressor.predict(x)

                elif regressor == "partialLSQ":

                    x -= np.mean(x, axis=0)
                    ux, sx, vx = np.linalg.svd(x, False)
                    optimal = min(ux.shape[1], 25)
                    b_aux, residuals[level][:, k] = _partial_least_squares(
                        x, y[:, k], ux, sx, vx.T, optimal, True
                    )

                # store results
                fit_mat[level][:, k] = b_aux

                if (k + 1) % 10 == 0 and self.verbose:
                    print(
                        "...%d/%d finished fitting..." % (k + 1, pcs.shape[1])
                    )

            if self.verbose:
                # check for negative definiteness
                negdef = {}
                for l, e, pos in zip(
                    fit_mat.keys()[::-1],
                    range(len(fit_mat.keys())),
                    range(len(fit_mat.keys()) - 1, -1, -1),
                ):
                    negdef[l] = fit_mat[l][
                        pos * pcs.shape[1] : (pos + 1) * pcs.shape[1]
                    ]
                    for a in range(pos - 1, -1, -1):
                        negdef[l] = np.c_[
                            negdef[l],
                            fit_mat[l][
                                a * pcs.shape[1] : (a + 1) * pcs.shape[1]
                            ],
                        ]
                    for a in range(e):
                        negdef[l] = np.c_[negdef[l], np.eye(pcs.shape[1])]
                grand_negdef = np.concatenate(
                    [negdef[a] for a in negdef.keys()], axis=0
                )
                d, _ = np.linalg.eig(grand_negdef)
                print("...maximum eigenvalue: %.4f" % (max(np.real(d))))

        self.residuals = residuals
        self.fit_mat = fit_mat

        if self.verbose:
            print("training done.")

    def integrate_model(
        self,
        n_realizations,
        int_length=None,
        noise_type="white",
        sigma=1.0,
        n_workers=3,
        diagnostics=True,
    ):
        """
        Integrate trained model.
        noise_type:
        -- white - classic white noise, spatial correlation by cov. matrix of last level residuals
        -- cond - find n_samples closest to the current space in subset of n_pcs and use their cov. matrix
        -- seasonal - seasonal dependence of the residuals, fit n_harm harmonics of annual cycle, could also be used with cond.
        except 'white', one can choose more settings like ['seasonal', 'cond']
        """

        if self.verbose:
            print("preparing to integrate model...")

        pcs = self.input_pcs.copy()
        pcs = pcs.T  # time x dim

        pcmax = np.amax(pcs, axis=0)
        pcmin = np.amin(pcs, axis=0)
        self.varpc = np.var(pcs, axis=0, ddof=1)

        self.int_length = pcs.shape[0] if int_length is None else int_length

        self.diagnostics = diagnostics

        if self.harmonic_pred in ["all", "first"]:
            if self.verbose:
                print("...using harmonic predictors (with annual frequency)...")
            self.xsin = np.sin(2 * np.pi * np.arange(self.int_length) / 12.0)
            self.xcos = np.cos(2 * np.pi * np.arange(self.int_length) / 12.0)

        if self.verbose:
            print("...preparing noise forcing...")

        self.sigma = sigma
        if isinstance(noise_type, str):
            if noise_type not in ["white", "cond", "seasonal"]:
                raise Exception(
                    "Unknown noise type to be used as forcing. Use 'white', 'cond', or 'seasonal'."
                )
        elif isinstance(noise_type, list):
            noise_type = frozenset(noise_type)
            if not noise_type.issubset(set(["white", "cond", "seasonal"])):
                raise Exception(
                    "Unknown noise type to be used as forcing. Use 'white', 'cond', or 'seasonal'."
                )

        self.last_level_res = self.residuals[max(self.residuals.keys())]
        self.noise_type = noise_type
        if noise_type == "white":
            if self.verbose:
                print("...using spatially correlated white noise...")
            Q = np.cov(self.last_level_res, rowvar=0)
            self.rr = np.linalg.cholesky(Q).T

        if "seasonal" in noise_type:
            n_harmonics = 5
            if self.verbose:
                print(
                    "...fitting %d harmonics to estimate seasonal modulation of last level's residual..."
                    % n_harmonics
                )
            if self.delay_model:
                resid_delayed = self.last_level_res[
                    -(self.last_level_res.shape[0] // 12) * 12 :
                ].copy()
                rr_last = np.reshape(
                    resid_delayed,
                    (
                        12,
                        self.last_level_res.shape[0] // 12,
                        self.last_level_res.shape[1],
                    ),
                    order="F",
                )
            else:
                rr_last = np.reshape(
                    self.last_level_res,
                    (
                        12,
                        self.last_level_res.shape[0] // 12,
                        self.last_level_res.shape[1],
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
            bamp = np.zeros((predictors.shape[1], pcs.shape[1]))
            for k in range(bamp.shape[1]):
                bamp[:, k] = np.linalg.lstsq(predictors, rr_last_std[:, k])[0]
            rr_last_std_ts = np.dot(predictors, bamp)
            self.rr_last_std_ts = np.repeat(
                rr_last_std_ts,
                repeats=self.last_level_res.shape[0] // 12,
                axis=0,
            )
            if self.delay_model:
                resid_delayed /= self.rr_last_std_ts
                Q = np.cov(resid_delayed, rowvar=0)
            else:
                self.last_level_res /= self.rr_last_std_ts
                Q = np.cov(self.last_level_res, rowvar=0)

            self.rr = np.linalg.cholesky(Q).T

        if diagnostics:
            if self.verbose:
                print("...running diagnostics for the data...")
            # ACF, kernel density, integral corr. timescale for data
            self.max_lag = 50
            lag_cors = np.zeros((2 * self.max_lag + 1, pcs.shape[1]))
            kernel_densities = np.zeros((100, pcs.shape[1], 2))
            for k in range(pcs.shape[1]):
                lag_cors[:, k] = cross_correlation(
                    pcs[:, k], pcs[:, k], max_lag=self.max_lag
                )
                (
                    kernel_densities[:, k, 0],
                    kernel_densities[:, k, 1],
                ) = kdensity_estimate(pcs[:, k], kernel="epanechnikov")
            integral_corr_timescale = np.sum(np.abs(lag_cors), axis=0)

            # init for integrations
            lag_cors_int = np.zeros([n_realizations] + list(lag_cors.shape))
            kernel_densities_int = np.zeros(
                [n_realizations] + list(kernel_densities.shape)
            )
            stat_moments_int = np.zeros(
                (4, n_realizations, pcs.shape[1])
            )  # mean, variance, skewness, kurtosis
            int_corr_scale_int = np.zeros((n_realizations, pcs.shape[1]))

        self.diagpc = np.diag(np.std(pcs, axis=0, ddof=1))
        self.maxpc = np.amax(np.abs(pcs))
        self.diagres = {}
        self.maxres = {}
        for l in self.residuals.keys():
            self.diagres[l] = np.diag(np.std(self.residuals[l], axis=0, ddof=1))
            self.maxres[l] = np.amax(np.abs(self.residuals[l]))

        self.pcs = pcs

        if n_workers > 1:
            # from multiprocessing import Pool
            from pathos.multiprocessing import ProcessingPool

            pool = ProcessingPool(n_workers)
            map_func = pool.amap
            if self.verbose:
                print(
                    "...running integration of %d realizations using %d workers..."
                    % (n_realizations, n_workers)
                )
        else:
            map_func = map
            if self.verbose:
                print(
                    "...running integration of %d realizations single threaded..."
                    % n_realizations
                )

        rnds = []
        for n in range(n_realizations):
            r = {}
            for l in self.fit_mat.keys():
                if l == 0:
                    if self.delay_model:
                        r[l] = np.dot(
                            self.diagpc,
                            np.random.normal(
                                0, sigma, (pcs.shape[1], self.delay)
                            ),
                        )
                    else:
                        r[l] = np.dot(
                            np.random.normal(0, sigma, (pcs.shape[1],)),
                            self.diagpc,
                        )
                else:
                    if self.delay_model:
                        r[l] = np.dot(
                            self.diagres[l - 1],
                            np.random.normal(
                                0, sigma, (pcs.shape[1], self.delay)
                            ),
                        )
                    else:
                        r[l] = np.dot(
                            np.random.normal(0, sigma, (pcs.shape[1],)),
                            self.diagres[l - 1],
                        )
            rnds.append(r)
        args = [
            [i, rnd, noise_type] for i, rnd in zip(range(n_realizations), rnds)
        ]
        results = map_func(self._process_integration, args)

        del args
        if n_workers > 1:
            pool.close()
            pool.join()

        self.integration_results = np.zeros(
            (n_realizations, pcs.shape[1], self.int_length)
        )
        self.num_exploding = np.zeros((n_realizations,))

        if n_workers > 1:
            results = results.get()

        if self.diagnostics:
            # x, num_exploding, xm, xv, xs, xk, lc, kden, ict
            for i, x, num_expl, xm, xv, xs, xk, lc, kden, ict in results:
                self.integration_results[i, ...] = x.T
                self.num_exploding[i] = num_expl
                stat_moments_int[0, i, :] = xm
                stat_moments_int[1, i, :] = xv
                stat_moments_int[2, i, :] = xs
                stat_moments_int[3, i, :] = xk
                lag_cors_int[i, ...] = lc
                kernel_densities_int[i, ...] = kden
                int_corr_scale_int[i, ...] = ict
        else:
            for i, x, num_expl in results:
                self.integration_results[i, ...] = x.T
                self.num_exploding[i] = num_expl

        if self.verbose:
            print("...integration done, now saving results...")

        if self.verbose:
            print("...results saved to structure.")
            print(
                "there was %d expolding integration chunks in %d realizations."
                % (np.sum(self.num_exploding), n_realizations)
            )

        if self.diagnostics:
            if self.verbose:
                print("plotting diagnostics...")

            import matplotlib.pyplot as plt

            # plot all diagnostic stuff
            ## mean, variance, skewness, kurtosis, integral corr. time scale
            tits = [
                "MEAN",
                "VARIANCE",
                "SKEWNESS",
                "KURTOSIS",
                "INTEGRAL CORRELATION TIME SCALE",
            ]
            plot = [
                np.mean(pcs, axis=0),
                np.var(pcs, axis=0, ddof=1),
                sts.skew(pcs, axis=0),
                sts.kurtosis(pcs, axis=0),
                integral_corr_timescale,
            ]
            xplot = np.arange(1, pcs.shape[1] + 1)
            for i, tit, p in zip(range(5), tits, plot):
                plt.figure()
                plt.title(tit, size=20)
                plt.plot(xplot, p, linewidth=3, color="#3E3436")
                if i < 4:
                    plt.plot(
                        xplot,
                        np.percentile(stat_moments_int[i, :, :], q=2.5, axis=0),
                        "--",
                        linewidth=2.5,
                        color="#EA3E36",
                    )
                    plt.plot(
                        xplot,
                        np.percentile(
                            stat_moments_int[i, :, :], q=97.5, axis=0
                        ),
                        "--",
                        linewidth=2.5,
                        color="#EA3E36",
                    )
                else:
                    plt.plot(
                        xplot,
                        np.percentile(int_corr_scale_int, q=2.5, axis=0),
                        "--",
                        linewidth=2.5,
                        color="#EA3E36",
                    )
                    plt.plot(
                        xplot,
                        np.percentile(int_corr_scale_int, q=97.5, axis=0),
                        "--",
                        linewidth=2.5,
                        color="#EA3E36",
                    )
                plt.xlabel("# PC", size=15)
                plt.xlim([xplot[0], xplot[-1]])
                plt.show()
                plt.close()

            ## lagged correlations, PDF - plot first 9 PCs (or less if input number of pcs is < 9)
            tits = ["AUTOCORRELATION", "PDF"]
            plot = [
                [lag_cors, lag_cors_int],
                [kernel_densities, kernel_densities_int],
            ]
            xlabs = ["LAG", ""]
            for i, tit, p, xlab in zip(range(2), tits, plot, xlabs):
                plt.figure()
                plt.suptitle(tit, size=25)
                no_plts = 9 if self.no_input_ts > 9 else self.no_input_ts
                for sub in range(0, no_plts):
                    plt.subplot(3, 3, sub + 1)
                    if i == 0:
                        xplt = np.arange(0, self.max_lag + 1)
                        plt.plot(
                            xplt,
                            p[0][p[0].shape[0] // 2 :, sub],
                            linewidth=3,
                            color="#3E3436",
                        )
                        plt.plot(
                            xplt,
                            np.percentile(
                                p[1][:, p[0].shape[0] // 2 :, sub],
                                q=2.5,
                                axis=0,
                            ),
                            "--",
                            linewidth=2.5,
                            color="#EA3E36",
                        )
                        plt.plot(
                            xplt,
                            np.percentile(
                                p[1][:, p[0].shape[0] // 2 :, sub],
                                q=97.5,
                                axis=0,
                            ),
                            "--",
                            linewidth=2.5,
                            color="#EA3E36",
                        )
                        plt.xlim([xplt[0], xplt[-1]])
                    else:
                        plt.plot(
                            p[0][:, sub, 0],
                            p[0][:, sub, 1],
                            linewidth=3,
                            color="#3E3436",
                        )
                        plt.plot(
                            p[1][0, :, sub, 0],
                            np.percentile(p[1][:, :, sub, 1], q=2.5, axis=0),
                            "--",
                            linewidth=2.5,
                            color="#EA3E36",
                        )
                        plt.plot(
                            p[1][0, :, sub, 0],
                            np.percentile(p[1][:, :, sub, 1], q=97.5, axis=0),
                            "--",
                            linewidth=2.5,
                            color="#EA3E36",
                        )
                        plt.xlim([p[0][0, sub, 0], p[0][-1, sub, 0]])
                    plt.xlabel(xlab, size=15)
                    plt.title("PC %d" % (int(sub) + 1), size=20)
                # plt.tight_layout()
                plt.show()
                plt.close()

    def _process_integration(self, a):

        i, rnd, noise = a
        num_exploding = 0
        repeats = 20
        xx = {}
        x = {}
        for l in self.fit_mat.keys():
            xx[l] = np.zeros((repeats, self.input_pcs.shape[0]))
            if self.delay_model:
                xx[l][: self.delay, :] = rnd[l].T
            else:
                xx[l][0, :] = rnd[l]

            x[l] = np.zeros((self.int_length, self.input_pcs.shape[0]))
            if self.delay_model:
                x[l][: self.delay, :] = xx[l][: self.delay, :]
            else:
                x[l][0, :] = xx[l][0, :]

        step0 = 0
        if self.delay_model:
            step = self.delay
        else:
            step = 1
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
                for l in self.fit_mat.keys():
                    zz[l] = xx[0][k - 1, :]
                    for lr in range(l):
                        zz[l] = np.r_[zz[l], xx[lr + 1][k - 1, :]]
                for l in self.fit_mat.keys():
                    if l == 0:
                        if self.quad:
                            q = np.tril(np.outer(zz[l].T, zz[l]), -1)
                            quad_pred = q[np.nonzero(q)]
                        if self.delay_model:
                            zz[l] = np.tanh(
                                self.kappa * x[l][step - self.delay, :]
                            )
                        if self.harmonic_pred in ["all", "first"]:
                            if self.quad:
                                zz[l] = np.r_[
                                    quad_pred,
                                    zz[l],
                                    zz[l] * self.xsin[step],
                                    zz[l] * self.xcos[step],
                                    self.xsin[step],
                                    self.xcos[step],
                                    1,
                                ]
                            else:
                                zz[l] = np.r_[
                                    zz[l],
                                    zz[l] * self.xsin[step],
                                    zz[l] * self.xcos[step],
                                    self.xsin[step],
                                    self.xcos[step],
                                    1,
                                ]
                        else:
                            if self.quad:
                                zz[l] = np.r_[quad_pred, zz[l], 1]
                            else:
                                zz[l] = np.r_[zz[l], 1]
                    else:
                        if self.harmonic_pred == "all":
                            zz[l] = np.r_[
                                zz[l],
                                zz[l] * self.xsin[step],
                                zz[l] * self.xcos[step],
                                self.xsin[step],
                                self.xcos[step],
                                1,
                            ]
                        else:
                            zz[l] = np.r_[zz[l], 1]

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
                        self.rr = np.linalg.cholesky(Q).T
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
                        self.rr = np.linalg.cholesky(Q).T

                # integration step
                for l in sorted(self.fit_mat, reverse=True):
                    if l == self.no_levels - 1:
                        forcing = np.dot(
                            self.rr,
                            np.random.normal(
                                0, self.sigma, (self.rr.shape[0],)
                            ).T,
                        )
                        if "seasonal" in self.noise_type:
                            forcing *= self.rr_last_std_ts[
                                step % self.rr_last_std_ts.shape[0], :
                            ]
                    else:
                        forcing = xx[l + 1][k, :]
                    xx[l][k, :] = (
                        xx[l][k - 1, :]
                        + np.dot(zz[l], self.fit_mat[l])
                        + forcing
                    )
                    # xx[l][k, :] = xx[l][k-1, :] + self.regressor.predict(zz[l]) + forcing

                step += 1

            # check if integration blows
            if np.amax(np.abs(xx[0])) <= 2 * self.maxpc and not np.any(
                np.isnan(xx[0])
            ):
                for l in self.fit_mat.keys():
                    x[l][step - repeats + 1 : step, :] = xx[l][1:, :]
                    # set first to last
                    xx[l][0, :] = xx[l][-1, :]
            else:
                for l in self.fit_mat.keys():
                    if l == 0:
                        xx[l][0, :] = np.dot(
                            np.random.normal(
                                0, self.sigma, (self.input_pcs.shape[0],)
                            ),
                            self.diagpc,
                        )
                    else:
                        xx[l][0, :] = np.dot(
                            np.random.normal(
                                0, self.sigma, (self.input_pcs.shape[0],)
                            ),
                            self.diagres[l - 1],
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
