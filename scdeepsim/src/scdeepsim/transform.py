import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


ArrayLike = Union[np.ndarray]


@dataclass
class AnalyticPearsonResidualScaler:
    """
    Analytic Pearson residual transformation for UMI count matrices.

    This is sklearn-StandardScaler-like:
    - fit() learns gene fractions p_g from a reference count matrix X.
    - transform() maps counts to analytic Pearson residuals using mu = n_c * p_g.
    - inverse_transform() approximately maps residuals back to counts using the stored mu.

    Notes
    -----
    1) This transformation is not truly invertible in general, especially if clipping is used.
       inverse_transform() performs an algebraic back-transform assuming mu is fixed.
    2) The returned "counts" from inverse_transform() are floats; you may choose to round them.
    3) X is expected to be nonnegative counts with shape (n_cells, n_genes).
    """

    theta: float = 100.0                 # theta=np.inf means Poisson
    clip: Optional[float] = None         # if None, default sqrt(n_cells) is used in transform
    eps: float = 1e-8                    # numerical stability
    store_mu_hat: bool = True            # store mu_hat_ for the last transformed data
    round_inverse: bool = False          # if True, round inverse counts to nearest int
    clip_inverse_nonneg: bool = True     # if True, clip inverse counts at 0

    # Learned / stored attributes (sklearn-style trailing underscore)
    p_g_: Optional[np.ndarray] = None          # shape (n_genes,)
    total_counts_: Optional[float] = None
    n_genes_: Optional[int] = None

    # Stored from the last fit/transform on some X
    n_counts_: Optional[np.ndarray] = None     # shape (n_cells,) for reference data
    mu_hat_: Optional[np.ndarray] = None       # shape (n_cells, n_genes) for last transformed X

    def fit(self, X: ArrayLike) -> "AnalyticPearsonResidualScaler":
        X = self._validate_X(X)
        n_cells, n_genes = X.shape
        col_sum = X.sum(axis=0)  # per gene
        total = float(col_sum.sum())

        if total <= 0:
            raise ValueError("Total counts must be positive to fit p_g.")

        # Gene fractions p_g
        p_g = col_sum / total
        # Avoid exact zeros (helpful for stability in downstream transforms)
        p_g = np.maximum(p_g, self.eps)
        p_g = p_g / p_g.sum()

        self.p_g_ = p_g.astype(np.float64, copy=False)
        self.total_counts_ = total
        self.n_genes_ = n_genes
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_X(X, n_genes=self.n_genes_)
        n_cells, _ = X.shape

        n_counts = X.sum(axis=1)  # per cell
        mu = n_counts[:, None] * self.p_g_[None, :]  # (n_cells, n_genes)

        denom = np.sqrt(mu + (mu * mu) / self.theta + self.eps) if np.isfinite(self.theta) else np.sqrt(mu + self.eps)
        Z = (X - mu) / denom

        # Default clipping as in scTransform/Lause et al.: +/- sqrt(n_cells)
        clip_val = self.clip if self.clip is not None else np.sqrt(n_cells)
        if np.isfinite(clip_val):
            Z = np.clip(Z, -clip_val, clip_val)

        # Store reference mu_hat for inverse_transform (last call)
        if self.store_mu_hat:
            self.n_counts_ = n_counts.astype(np.float64, copy=False)
            self.mu_hat_ = mu.astype(np.float64, copy=False)

        return Z.astype(np.float32, copy=False)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: ArrayLike, n_counts: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Approximate inverse mapping from residuals back to counts.

        Parameters
        ----------
        Z : array-like, shape (n_cells, n_genes)
            Pearson residuals.
        n_counts : optional, shape (n_cells,)
            If provided, uses mu = n_counts * p_g.
            If not provided, will use stored n_counts_ from the most recent transform
            (only valid when inverting residuals that correspond to that same data).

        Returns
        -------
        X_rec : np.ndarray, shape (n_cells, n_genes)
            Reconstructed (float) counts.
        """
        self._check_is_fitted()
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array (n_cells, n_genes).")
        if self.n_genes_ is not None and Z.shape[1] != self.n_genes_:
            raise ValueError(f"Z has {Z.shape[1]} genes but model was fit with {self.n_genes_} genes.")

        if n_counts is None:
            if self.n_counts_ is None:
                raise ValueError(
                    "n_counts is required because no reference n_counts_ is stored. "
                    "Call transform()/fit_transform() first (with store_mu_hat=True) "
                    "or pass n_counts explicitly."
                )
            n_counts = self.n_counts_
        else:
            n_counts = np.asarray(n_counts, dtype=np.float64)
            if n_counts.ndim != 1 or n_counts.shape[0] != Z.shape[0]:
                raise ValueError("n_counts must be a 1D array with length equal to n_cells in Z.")

        mu = n_counts[:, None] * self.p_g_[None, :]

        denom = np.sqrt(mu + (mu * mu) / self.theta + self.eps) if np.isfinite(self.theta) else np.sqrt(mu + self.eps)
        X_rec = Z * denom + mu

        if self.clip_inverse_nonneg:
            X_rec = np.maximum(X_rec, 0.0)

        if self.round_inverse:
            X_rec = np.rint(X_rec)

        return X_rec.astype(np.float32, copy=False)

    # --------- helpers ---------
    def _check_is_fitted(self) -> None:
        if self.p_g_ is None or self.n_genes_ is None:
            raise RuntimeError("This AnalyticPearsonResidualScaler instance is not fitted yet. Call fit() first.")

    def _validate_X(self, X: ArrayLike, n_genes: Optional[int] = None) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_cells, n_genes).")
        if n_genes is not None and X.shape[1] != n_genes:
            raise ValueError(f"X has {X.shape[1]} genes but expected {n_genes}.")
        if np.any(X < 0):
            raise ValueError("X must be nonnegative counts.")
        # Use float for safe arithmetic; keep sparse out-of-scope for this minimal class
        return X.astype(np.float64, copy=False)



class NormalLog1pScaler:
    """
    Normalize per cell to `target_sum` and apply log1p.
    Stores per-cell row sums from `fit()` to enable a pseudo-inverse.

    Note: Only valid for matrices with the same number/order of cells as used in fit().
    """

    def __init__(self, target_sum: float = 1e4, eps: float = 1e-12):
        self.target_sum = float(target_sum)
        self.eps = float(eps)
        self.row_sum: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike) -> "NormalLog1pScaler":
        X = self._validate_X(X)
        rs = X.sum(axis=1).astype(np.float64, copy=False)
        self.row_sum = np.maximum(rs, self.eps)  # avoid divide-by-zero
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_X(X, n_cells=self.row_sum.shape[0])
        return np.log1p((X / self.row_sum[:, None]) * self.target_sum)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(
        self,
        X: ArrayLike,
        clip: bool = True,
        do_round: bool = True,
        return_int: bool = False,
    ) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_X(X, n_cells=self.row_sum.shape[0])
        X = np.expm1(X) * (self.row_sum[:, None] / self.target_sum)
        if clip:
            X = np.clip(X, 0, np.inf)
        if do_round:
            X = np.rint(X)
        if return_int:
            return X.astype(np.int64, copy=False)
        return X.astype(np.float32, copy=False)

    def _check_is_fitted(self) -> None:
        if self.row_sum is None:
            raise RuntimeError("This NormalLog1pScaler instance is not fitted yet. Call fit() first.")

    def _validate_X(
        self,
        X: ArrayLike,
        n_genes: Optional[int] = None,
        n_cells: Optional[int] = None,
    ) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array (n_cells, n_genes).")
        if n_cells is not None and X.shape[0] != n_cells:
            raise ValueError(f"X has {X.shape[0]} cells but expected {n_cells}.")
        if n_genes is not None and X.shape[1] != n_genes:
            raise ValueError(f"X has {X.shape[1]} genes but expected {n_genes}.")
        if np.any(X < 0):
            raise ValueError("X must be nonnegative.")
        return X.astype(np.float64, copy=False)