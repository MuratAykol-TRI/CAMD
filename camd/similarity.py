import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM
from scipy.spatial.distance import cdist
from pymatgen import Composition
from camd.agent.agents import diverse_quant
from joblib import Parallel, delayed


class FunctionalSimilarity:
    def __init__(self, df, curated_ids, scale=True, pca=None, pca_kernel=None):
        """
        FunctionalSimilarity incorporates a "similarity" based mining approach for the
        difficult problem of finding materials that possess a certain functionality
        that is either difficult or expensive to quantify using experiments. The class
        requires a data frame of features, where the index is the material label,
        and a curated list of labels provided by the user as examples of materials
        that possess the targeted functionality. Current distance/similarity methods available:
            - ['dice', 'correlation', 'cityblock',  'cosine', 'mahalanobis', 'euclidean', 'tanimoto']
        If similarity measure needs to be derived from a distance, we use:
            - similarity = 1/(1+distance)
        For general use cases, a filtered and ranked version of the initial data frame of materials can be accessed
            using the get_df_of_similar method. For only getting a similarity-ranked list of material labels,
            get_ranked_ids method can be used.
        If there are more than a few example materials in curated list, FunctionalSimilarity provides
            autofind_best_metric, a cross-validation type approach to automatically finding the most suitable
            similarity metric.

        Args:
            df (pandas.DataFrame): features of materials to search over. index labels are interpreted as unique
                ids for materials.
            curated_ids (list): labels of materials in df that are known to deliver the target functionality.
                (e.g. known battery electrodes, superconductors, magneticaloric materials, thermoelectrics, etc.)
            scale (bool): whether the df should be standardized. defaults to True. Scaling df a priori can speed up
                autofind methods.
            pca (int): if provided, input features will be PCA transformed and this many first principal
                components will be used in similarity / distance measurements. defaults to None, which means
                no pca transformation will be done on data.
        """
        self._df = df
        self._curated_ids = np.array(curated_ids)
        if scale:
            self.scaler = StandardScaler()
            self._X = self.scaler.fit_transform(
                df.drop(["Composition", "similarities"], axis=1, errors="ignore")
            )
            self._X_scaled = True
        else:
            self._X = self._df.drop(
                ["Composition", "similarities"], axis=1, errors="ignore"
            ).to_numpy()
            self._X_scaled = False

        self._pca = False
        if pca:
            if pca_kernel:
                self._X = kernelpca(self._X, n_components=pca, kernel=pca_kernel)
                self._pca_kernel = pca_kernel
            else:
                _pca = PCA(n_components=pca)
                self._X = _pca.fit_transform(self._X)
                self._pca_kernel = None
            self._pca = pca

        self._metrics_allowed = [
            "dice",
            "correlation",
            "cityblock",
            "cosine",
            "mahalanobis",
            "euclidean",
            "tanimoto",
            "svm",
        ]
        self.similarities = {}
        self._rank = {}
        self._counts = {}
        self.best_metric = None
        self.use_features = None

    @property
    def X(self):
        if self.use_features:
            return self._X[:, self.use_features]
        else:
            return self._X

    @property
    def curated_ids(self):
        return self._curated_ids

    @property
    def curated_ilocs(self):
        return [self._df.index.get_loc(i) for i in self.curated_ids]

    @property
    def metrics_allowed(self):
        return self._metrics_allowed

    def _validate_metric(self, metric):
        return metric in self._metrics_allowed

    def compute_metric(self, metric="euclidean", metric_params=None):
        """
        Computes the similarity with the given metric. Results are stored in similarities attribute.
        Args:
            metric (str): one of the allowed similarity metrics.
        """
        if not self._validate_metric(metric):
            raise ValueError("not a valid metric.")

        if metric == "mahalanobis":
            self.similarities["mahalanobis"] = self.mahalanobis()

        elif metric == "tanimoto":
            self.similarities["tanimoto"] = self.tanimoto_dice(metric)

        elif metric == "dice":
            self.similarities["dice"] = self.tanimoto_dice(metric)

        elif metric == "svm":
            params = {}
            if metric_params:
                if "svm" in metric_params:
                    params = metric_params["svm"]
            self.similarities["svm"] = self.svm(**params)
        else:
            distances = cdist(self.X, self.X[self.curated_ilocs], metric=metric)
            self.similarities[metric] = (1.0 + distances) ** -1

    def get_ranked_ids(self, metric="euclidean", metric_params=None):
        """
        Args:
            metric (str): one of the allowed similarity metrics
        Returns:
            A ranked list of labels corresponding from original material df provided.
        """
        if metric not in self.similarities:
            self.compute_metric(metric, metric_params)
        return self._df.iloc[
            np.argsort(-np.mean(self.similarities[metric], axis=1))
        ].index.to_list()

    def get_df_of_similar(
        self,
        metric="euclidean",
        remove_curated=True,
        ignore_elements=None,
        include_elements=None,
        diversify=0,
    ):
        """
        Main method for getting a ranked version of df based on the similarity metric asked.
        Args:
            metric (str): one of the allowed similarity metrics
            remove_curated (bool): whether the curated ids should be removed from returned df
            ignore_elements (list): element strings to remove from returned df
            include_elements (list): element strings to include in returned df
            diversify (bool): diversification method from camd agents.
        Returns:
            a ranked and filter version of data frame of materials.
        """
        _result = self._df.loc[self.get_ranked_ids(metric)]
        if remove_curated:
            _result = _result.drop(self.curated_ids)

        if ignore_elements or include_elements:
            ignore_compound_labels = []
            ignore_elements = set(ignore_elements) if ignore_elements else set([-1])
            include_elements = set(include_elements) if include_elements else set()
            for r in _result.iterrows():
                c = set(Composition(r[1]["Composition"]).as_dict().keys())
                if ignore_elements.issubset(c):
                    ignore_compound_labels.append(r[0])
                    continue
                if not include_elements.issubset(c):
                    ignore_compound_labels.append(r[0])
            _result = _result.drop(ignore_compound_labels)

        if diversify:
            diverse_quant_ids = diverse_quant(
                _result.index.tolist()[: diversify * 500],
                target_length=diversify,
                df=_result.drop("similarities", axis=1, errors="ignore")[
                    : diversify * 500
                ],
            )
            _result = _result.loc[diverse_quant_ids]
        return _result

    def mahalanobis(
        self, pca_components=50, pca_sub_metric="euclidean", pca_mah_scale=True
    ):
        if self._pca:
            pca_components = min(pca_components, self._pca, self.X.shape[1])
        pca = PCA(n_components=pca_components)
        X = pca.fit_transform(self.X)
        if pca_mah_scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        distances = cdist(X, X[self.curated_ilocs], metric=pca_sub_metric)
        similarities = (1.0 + distances) ** -1
        return similarities

    def tanimoto_dice(self, mode="tanimoto"):
        similarities = []
        for x in self.X:
            _similarities = []
            for j in self.X[self.curated_ilocs]:
                _similarities.append(self._tanimoto_dice(x, j, mode=mode))
            similarities.append(_similarities)
        return np.array(similarities)

    @staticmethod
    def _tanimoto_dice(A, B, mode="tanimoto"):
        dot = np.dot(A, B)
        if mode == "tanimoto":
            return dot / (np.sum(A ** 2) + np.sum(B ** 2) - dot)
        elif mode == "dice":
            return 2 * dot / (np.sum(A ** 2) + np.sum(B ** 2))
        else:
            raise ValueError("no such mode")

    def svm(self, **kwargs):
        if kwargs:
            kernel = kwargs.get("kernel", "rbf")
        else:
            kernel = "rbf"
        clf = OneClassSVM(gamma="scale", kernel=kernel).fit(self.X[self.curated_ilocs])
        scores = clf.score_samples(self.X)
        return np.vstack((scores, scores)).T

    def _get_metric_ranks(
        self, n_splits=5, random_state=42, repeats=1, metrics=None, metric_params=None
    ):
        if metrics:
            if not np.alltrue([self._validate_metric(i) for i in metrics]):
                raise ValueError("invalid metric found")
        else:
            metrics = self._metrics_allowed
        ranks = dict([(m, []) for m in metrics])

        print("iterations ", repeats * n_splits * len(metrics), ":")
        for _ in range(repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state * _)
            for test_index, train_index in kf.split(self.curated_ids):
                fn = FunctionalSimilarity(
                    self._df, self.curated_ids[train_index], scale=self._X_scaled
                )
                fn._X = self._X
                fn._pca = self._pca
                fn.use_features = self.use_features

                for metric in metrics:
                    print(".", end="")
                    ranked_ids = fn.get_ranked_ids(metric, metric_params)
                    for i in self.curated_ids[test_index]:
                        ranks[metric].append(ranked_ids.index(i))
        self._ranks = ranks
        return self._ranks

    def autofind_best_metric(
        self,
        n_splits=5,
        random_state=42,
        repeats=1,
        stop=10000,
        num=1001,
        metrics=None,
        metric_params=None,
    ):
        """
        Method to find the best similarity metric via cross-validation over area under
        cumulative recall curves.
        Args:
            n_splits (int): number of splits in k-fold
            random_state (int): seed for random state
            repeats (int): repeats the random k-fold this many times
            stop (int): upper limit for ranking scan for AUC calculation
            num (int): number of points considered in ranking scan for AUC calculation
            metrics (list): similarity metrics to consider. defaults to None, which will use all allowed metrics.
        """
        self._get_metric_ranks(n_splits, random_state, repeats, metrics, metric_params)
        self._top = np.linspace(0, stop, num, dtype=int)
        self._counts = {}

        for metric in self._ranks:
            c = []
            for i in self._top[1:]:
                p = (
                    sum(np.array(self._ranks[metric]) < i)
                    / len(self._ranks[metric])
                    * 100
                )
                c.append(p)
            self._counts[metric] = c
        self._areas = {}
        for metric in self._counts:
            d = np.array([0] + self._counts[metric])
            self._areas[metric] = np.sum(
                (d[:-1] + d[1:]) * (self._top[1:] - self._top[:-1]) / 2.0
            )
        self.best_metirc = sorted(
            list(self._areas.items()), key=lambda x: x[1], reverse=True
        )[0][0]
        return self.best_metirc

    def plot_auto_ranks(self, plt_obj=None):
        """
        Plots the CV-based cumulative recalls for auto finding best metric.
        """
        if not self._counts:
            raise ValueError("Need to run autofind_best_metric.")
        _plt = plt_obj if plt_obj else plt
        for metric in self._counts:
            _plt.plot(
                self._top,
                [0] + self._counts[metric],
                "-",
                label=metric,
                linewidth=2,
                alpha=1,
            )
        _plt.ylabel("Cumulative recall (%)", fontsize=12)
        _plt.xlabel("Number of tests", fontsize=12)
        _plt.minorticks_on()
        _plt.legend(frameon=False)
        _plt.ylim(0,)
        _plt.xlim(0,)
        return _plt


def kernelpca(X, n_components=None, kernel=None):
    _pca = KernelPCA(n_components=n_components, kernel=kernel)
    if X.shape[0] > 10000:
        _pca.fit(X[np.random.choice(X.shape[0], size=10000, replace=False)])
        X = Parallel(n_jobs=-1, verbose=1)(
            delayed(_pca.transform)(i.reshape(1, -1)) for i in X
        )
        X = np.array(X)
        X = X.reshape(X.shape[0], X.shape[2])
    else:
        X = _pca.fit_transform(X)
    return X
