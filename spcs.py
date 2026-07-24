#!/usr/bin/env python3
"""SPCS smoothing for spatial transcriptomics in AnnData (.h5ad) format."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


EPS = 1e-8


@dataclass
class NeighborSet:
    indices: np.ndarray
    weights: np.ndarray


@dataclass
class GeneFilterResult:
    expression: np.ndarray
    keep_idx: np.ndarray


def _to_dense(matrix: np.ndarray | sp.spmatrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _extract_expression(adata: ad.AnnData, layer: Optional[str]) -> np.ndarray:
    if layer is None:
        return _to_dense(adata.X)
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    return _to_dense(adata.layers[layer])


def _filter_genes(
    expr_sg: np.ndarray,
    zero_cutoff: float,
    var_cutoff: float,
) -> GeneFilterResult:
    if not (0.0 <= zero_cutoff <= 1.0):
        raise ValueError("gene_zero_cutoff must be in [0, 1].")
    if not (0.0 <= var_cutoff < 1.0):
        raise ValueError("gene_var_cutoff must be in [0, 1).")

    zero_frac = np.mean(expr_sg == 0.0, axis=0)
    keep_zero = zero_frac <= zero_cutoff
    if not np.any(keep_zero):
        raise ValueError("Gene filtering removed all genes at zero-cutoff step.")

    filtered = expr_sg[:, keep_zero]
    if var_cutoff > 0.0 and filtered.shape[1] > 1:
        vars_ = np.var(filtered, axis=0)
        thresh = np.quantile(vars_, var_cutoff)
        keep_var = vars_ >= thresh
        if not np.any(keep_var):
            raise ValueError("Gene filtering removed all genes at variance-cutoff step.")
        keep_idx = np.flatnonzero(keep_zero)[keep_var]
    else:
        keep_idx = np.flatnonzero(keep_zero)

    return GeneFilterResult(expression=expr_sg[:, keep_idx], keep_idx=keep_idx)


def _log_cpm(expr_sg: np.ndarray, scale: float = 1e6) -> np.ndarray:
    if scale <= 0:
        raise ValueError("CPM scale must be positive.")
    lib = expr_sg.sum(axis=1, keepdims=True)
    lib = np.where(lib <= 0.0, 1.0, lib)
    cpm = (expr_sg / lib) * scale
    return np.log2(cpm + 1.0)


def _extract_coordinates(
    adata: ad.AnnData,
    x_key: Optional[str],
    y_key: Optional[str],
) -> np.ndarray:
    if (x_key is None) != (y_key is None):
        raise ValueError("Provide both x_key and y_key, or neither.")
    if x_key is not None and y_key is not None:
        if x_key not in adata.obs or y_key not in adata.obs:
            raise ValueError(f"obs keys '{x_key}' and/or '{y_key}' not found.")
        return adata.obs[[x_key, y_key]].to_numpy(dtype=float)

    candidates = [("array_row", "array_col"), ("st.x", "st.y"), ("coord1", "coord2")]
    for ckx, cky in candidates:
        if ckx in adata.obs and cky in adata.obs:
            return adata.obs[[ckx, cky]].to_numpy(dtype=float)

    if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2:
        return np.asarray(adata.obsm["spatial"][:, :2], dtype=float)

    raise ValueError(
        "Unable to infer spatial coordinates. Provide --x-key/--y-key, add "
        "obs[['array_row','array_col']], or populate obsm['spatial']."
    )


def _is_integer_grid(coords: np.ndarray) -> bool:
    return np.all(np.isclose(coords, np.rint(coords)))


def _calc_pattern_contribution(spot_by_gene: np.ndarray, n_pcs: int) -> np.ndarray:
    n_components = min(n_pcs, spot_by_gene.shape[0], spot_by_gene.shape[1])
    if n_components < 1:
        raise ValueError("Expression matrix is empty.")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    pca_embedding = pca.fit_transform(spot_by_gene)
    p_dist = pairwise_distances(pca_embedding, metric="correlation")
    p_dist = np.nan_to_num(p_dist, nan=2.0, posinf=2.0, neginf=2.0)

    p_contrib = np.exp(-4.5 * np.square(p_dist))
    p_contrib[p_dist > 1.0] = 0.0
    np.fill_diagonal(p_contrib, 0.0)
    return p_contrib


def _find_pattern_neighbors(p_contrib: np.ndarray, tau_p: int) -> list[Optional[NeighborSet]]:
    if tau_p < 0:
        raise ValueError("tau_p must be >= 0.")

    n_spots = p_contrib.shape[0]
    neighbors: list[Optional[NeighborSet]] = []
    for i in range(n_spots):
        idx = np.flatnonzero(p_contrib[:, i] > 0.0)
        if idx.size == 0:
            neighbors.append(None)
            continue

        contrib = p_contrib[idx, i]
        if tau_p > 0 and idx.size > tau_p:
            top = np.argpartition(-contrib, tau_p - 1)[:tau_p]
            idx = idx[top]
            contrib = contrib[top]

        weights = contrib / (contrib.sum() + EPS)
        neighbors.append(NeighborSet(indices=idx, weights=weights))
    return neighbors


def _offset_points(order: int) -> np.ndarray:
    offsets = []
    for dx in range(-order, order + 1):
        for dy in range(-order, order + 1):
            dist = abs(dx) + abs(dy)
            if 0 < dist <= order:
                offsets.append((dx, dy, dist))
    return np.asarray(offsets, dtype=float)


def _find_spatial_neighbors(
    coords: np.ndarray,
    tau_s: int,
    is_hexa: bool,
) -> list[Optional[NeighborSet]]:
    if tau_s < 0:
        raise ValueError("tau_s must be >= 0.")
    order = tau_s * 2 if is_hexa else tau_s
    if order == 0:
        return [None] * coords.shape[0]

    if not _is_integer_grid(coords):
        l1 = pairwise_distances(coords, metric="manhattan")
        np.fill_diagonal(l1, np.inf)
        neighbors: list[Optional[NeighborSet]] = []
        for i in range(coords.shape[0]):
            idx = np.flatnonzero(l1[i] <= order)
            if idx.size == 0:
                neighbors.append(None)
                continue
            contrib = 1.0 / l1[i, idx]
            weights = contrib / (contrib.sum() + EPS)
            neighbors.append(NeighborSet(indices=idx, weights=weights))
        return neighbors

    offsets = _offset_points(order)
    idx_by_coord = {tuple(c): i for i, c in enumerate(coords.tolist())}

    neighbors: list[Optional[NeighborSet]] = []
    for c in coords:
        members = []
        weights = []
        for dx, dy, dist in offsets:
            j = idx_by_coord.get((c[0] + dx, c[1] + dy))
            if j is not None:
                members.append(j)
                weights.append(1.0 / dist)

        if not members:
            neighbors.append(None)
            continue

        w = np.asarray(weights, dtype=float)
        w = w / (w.sum() + EPS)
        neighbors.append(NeighborSet(indices=np.asarray(members, dtype=int), weights=w))
    return neighbors


def _smooth_expression(
    expr_gs: np.ndarray,
    s_neighbors: list[Optional[NeighborSet]],
    p_neighbors: list[Optional[NeighborSet]],
    alpha: float,
    beta: float,
) -> np.ndarray:
    n_genes, n_spots = expr_gs.shape
    s_exp = np.zeros((n_genes, n_spots), dtype=float)
    p_exp = np.zeros((n_genes, n_spots), dtype=float)

    for i in range(n_spots):
        s_nb = s_neighbors[i]
        if s_nb is not None:
            s_exp[:, i] = expr_gs[:, s_nb.indices] @ s_nb.weights

        p_nb = p_neighbors[i]
        if p_nb is not None:
            p_exp[:, i] = expr_gs[:, p_nb.indices] @ p_nb.weights

    return expr_gs * (1.0 - alpha) + (s_exp * beta + p_exp * (1.0 - beta)) * alpha


def _fill_blanks(
    expr_gs: np.ndarray,
    coords: np.ndarray,
    tau_s: int,
    is_hexa: bool,
    filling_thres: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    order = tau_s * 2 if is_hexa else tau_s
    if order == 0:
        return np.zeros((expr_gs.shape[0], 0), dtype=float), pd.DataFrame(columns=["st.x", "st.y"])
    if not _is_integer_grid(coords):
        raise ValueError("Padding requires integer grid coordinates. Disable padding or supply grid coordinates.")

    coords_int = np.rint(coords).astype(int)
    x_min, y_min = coords_int.min(axis=0)
    x_max, y_max = coords_int.max(axis=0)
    n_grid = (x_max - x_min + 1) * (y_max - y_min + 1)
    if n_grid > 2_000_000:
        raise ValueError(
            "Padding grid is too large for dense enumeration. Provide array_row/array_col style coordinates "
            "or disable padding."
        )

    full_grid = np.array([(x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)], dtype=int)
    if is_hexa:
        keep = (full_grid[:, 0] % 2) == (full_grid[:, 1] % 2)
        full_grid = full_grid[keep]

    existing = {tuple(c) for c in coords_int.tolist()}
    missing = np.array([pt for pt in full_grid.tolist() if tuple(pt) not in existing], dtype=int)
    if missing.size == 0:
        return np.zeros((expr_gs.shape[0], 0), dtype=float), pd.DataFrame(columns=["st.x", "st.y"])

    coord_to_idx = {tuple(c): i for i, c in enumerate(coords_int.tolist())}
    offsets = _offset_points(order)
    if is_hexa:
        offsets = offsets[(offsets[:, 2] % 2) == 0]

    filled_vectors = []
    filled_rows = []
    supp_counter = 1
    for x, y in missing:
        n_idx = []
        n_w = []
        for dx, dy, dist in offsets:
            maybe = (int(x + dx), int(y + dy))
            j = coord_to_idx.get(maybe)
            if j is not None:
                n_idx.append(j)
                n_w.append(1.0 / dist)

        if not n_idx:
            continue
        if len(n_idx) <= filling_thres * len(offsets):
            continue

        w = np.asarray(n_w, dtype=float)
        w = w / (w.sum() + EPS)
        filled_vectors.append(expr_gs[:, np.asarray(n_idx, dtype=int)] @ w)
        filled_rows.append({"st.x": int(x), "st.y": int(y), "barcodes": f"SUPP{supp_counter}"})
        supp_counter += 1

    if not filled_vectors:
        return np.zeros((expr_gs.shape[0], 0), dtype=float), pd.DataFrame(columns=["st.x", "st.y", "barcodes"])

    return np.column_stack(filled_vectors), pd.DataFrame(filled_rows)


def spcs_smooth_adata(
    adata: ad.AnnData,
    *,
    layer: Optional[str] = None,
    x_key: Optional[str] = None,
    y_key: Optional[str] = None,
    gene_zero_cutoff: float = 0.7,
    gene_var_cutoff: float = 0.0,
    cpm_scale: float = 1e6,
    tau_p: int = 16,
    tau_s: int = 2,
    alpha: float = 0.6,
    beta: float = 0.4,
    n_pcs: int = 10,
    is_hexa: bool = False,
    is_padding: bool = False,
    filling_thres: float = 0.5,
    output_layer: str = "spcs",
    inplace: bool = False,
) -> ad.AnnData:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    target = adata if inplace else adata.copy()
    expr = _extract_expression(target, layer)  # spots x genes
    filtered = _filter_genes(expr, zero_cutoff=gene_zero_cutoff, var_cutoff=gene_var_cutoff)
    kept_names = target.var_names[filtered.keep_idx].copy()
    target = target[:, filtered.keep_idx].copy()
    target.var["spcs_kept"] = True
    target.uns["spcs_filtering"] = {
        "gene_zero_cutoff": float(gene_zero_cutoff),
        "gene_var_cutoff": float(gene_var_cutoff),
        "n_genes_input": int(adata.n_vars),
        "n_genes_kept": int(len(filtered.keep_idx)),
        "kept_gene_names": kept_names.to_list(),
    }
    expr = _log_cpm(filtered.expression, scale=cpm_scale)
    coords = _extract_coordinates(target, x_key, y_key)

    expr_gs = expr.T.astype(float, copy=False)  # genes x spots
    p_contrib = _calc_pattern_contribution(expr, n_pcs=n_pcs)
    p_neighbors = _find_pattern_neighbors(p_contrib, tau_p=tau_p)
    s_neighbors = _find_spatial_neighbors(coords, tau_s=tau_s, is_hexa=is_hexa)

    smoothed_gs = _smooth_expression(expr_gs, s_neighbors, p_neighbors, alpha=alpha, beta=beta)
    smoothed_gs = np.nan_to_num(smoothed_gs, nan=np.nanmean(smoothed_gs))
    smoothed = smoothed_gs.T

    if is_padding:
        filled_expr_gs, filled_meta = _fill_blanks(
            smoothed_gs,
            coords,
            tau_s=tau_s,
            is_hexa=is_hexa,
            filling_thres=filling_thres,
        )
        if filled_expr_gs.shape[1] > 0:
            filled = filled_expr_gs.T
            target.X = smoothed
            target.layers[output_layer] = smoothed

            pad_obs = pd.DataFrame(index=filled_meta["barcodes"].tolist())
            for col in target.obs.columns:
                pad_obs[col] = np.nan
            pad_obs["spcs_is_padding"] = True
            if "spcs_is_padding" not in target.obs.columns:
                target.obs["spcs_is_padding"] = False
            target.obs["spcs_is_padding"] = target.obs["spcs_is_padding"].astype(bool)

            pad_adata = ad.AnnData(X=filled, obs=pad_obs, var=target.var.copy())
            if "spatial" in target.obsm:
                spatial = np.asarray(target.obsm["spatial"], dtype=float)
                supp_xy = filled_meta[["st.x", "st.y"]].to_numpy(dtype=float)
                supp_spatial = np.full((supp_xy.shape[0], spatial.shape[1]), np.nan, dtype=float)
                supp_spatial[:, :2] = supp_xy
                pad_adata.obsm["spatial"] = supp_spatial

            target = ad.concat([target, pad_adata], axis=0, merge="same", join="outer")
            target.layers[output_layer] = target.X.copy()
        else:
            target.layers[output_layer] = smoothed
            return target

        return target

    target.layers[output_layer] = smoothed
    return target


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPCS smoothing for h5ad spatial transcriptomics data.")
    parser.add_argument("--input", required=True, help="Input .h5ad path.")
    parser.add_argument("--output", required=True, help="Output .h5ad path.")
    parser.add_argument("--layer", default=None, help="Input expression layer. Default: adata.X")
    parser.add_argument("--x-key", default=None, help="obs key for x coordinate.")
    parser.add_argument("--y-key", default=None, help="obs key for y coordinate.")
    parser.add_argument(
        "--gene-zero-cutoff",
        type=float,
        default=1.0,
        help="Keep genes with <= this zero-expression fraction.",
    )
    parser.add_argument(
        "--gene-var-cutoff",
        type=float,
        default=0.0,
        help="Drop genes below this variance quantile after zero filtering.",
    )
    parser.add_argument(
        "--cpm-scale",
        type=float,
        default=1e6,
        help="CPM scaling factor used before mandatory log2(CPM+1).",
    )
    parser.add_argument("--tau-p", type=int, default=16, help="Pattern-neighbor count (0 means all positive-contrib).")
    parser.add_argument("--tau-s", type=int, default=2, help="Spatial neighborhood order.")
    parser.add_argument("--alpha", type=float, default=0.6, help="SPCS alpha in [0,1].")
    parser.add_argument("--beta", type=float, default=0.4, help="SPCS beta in [0,1].")
    parser.add_argument("--n-pcs", type=int, default=10, help="PC dimensions used for pattern space.")
    parser.add_argument("--is-hexa", action="store_true", help="Use Visium/hex grid mode (doubles spatial order).")
    parser.add_argument("--is-padding", action="store_true", help="Fill missing grid spots.")
    parser.add_argument("--filling-thres", type=float, default=0.5, help="Minimum occupied-neighbor ratio for padding.")
    parser.add_argument("--output-layer", default="spcs", help="Layer name to store smoothed matrix.")
    parser.add_argument(
        "--replace-x",
        action="store_true",
        help="Replace adata.X with smoothed expression (still also writes to output layer).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    adata = ad.read_h5ad(args.input)
    out = spcs_smooth_adata(
        adata,
        layer=args.layer,
        x_key=args.x_key,
        y_key=args.y_key,
        gene_zero_cutoff=args.gene_zero_cutoff,
        gene_var_cutoff=args.gene_var_cutoff,
        cpm_scale=args.cpm_scale,
        tau_p=args.tau_p,
        tau_s=args.tau_s,
        alpha=args.alpha,
        beta=args.beta,
        n_pcs=args.n_pcs,
        is_hexa=args.is_hexa,
        is_padding=args.is_padding,
        filling_thres=args.filling_thres,
        output_layer=args.output_layer,
        inplace=False,
    )
    if args.replace_x:
        out.X = out.layers[args.output_layer].copy()
    out.write_h5ad(args.output)


if __name__ == "__main__":
    main()
