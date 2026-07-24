#!/usr/bin/env bash
# Demo of every spcs.py parameter. Replace the placeholder values below with
# your own paths and settings; delete any flag you don't need.

python spcs.py \
    --input /path/to/input.h5ad \
    --output /path/to/output.h5ad \
    --layer raw_counts \
    --x-key "he_x" \
    --y-key "he_y" \
    --gene-zero-cutoff 0.7 \
    --gene-var-cutoff 0.1 \
    --cpm-scale 1000000 \
    --tau-p 16 \
    --tau-s 2 \
    --alpha 0.6 \
    --beta 0.4 \
    --n-pcs 10 \
    --is-hexa \
    --is-padding \
    --filling-thres 0.5 \
    --output-layer spcs \
    --replace-x

# --input            (required) Path to the input .h5ad file.
# --output           (required) Path to write the smoothed .h5ad file.
# --layer            Expression layer in adata.layers to smooth. Omit to use adata.X.
# --x-key / --y-key  obs column names holding spatial x/y coordinates. Omit both to
#                    auto-detect from obs[['array_row','array_col']], obs[['st.x','st.y']],
#                    obs[['coord1','coord2']], or obsm['spatial'].
# --gene-zero-cutoff Keep genes whose fraction of zero-expression spots is <= this value (0-1).
# --gene-var-cutoff  Drop genes below this variance quantile after zero filtering (0-1, exclusive of 1).
# --cpm-scale        Scaling factor for CPM normalization before log2(CPM+1) (e.g. 1e6).
# --tau-p            Number of pattern-space (expression) neighbors to use; 0 = all with positive contribution.
# --tau-s            Spatial neighborhood order (how many rings/steps out to consider).
# --alpha            Overall smoothing strength in [0,1]; 0 = no smoothing, 1 = fully replaced by neighbor blend.
# --beta             Balance between spatial and pattern neighbors in [0,1]; 1 = spatial only, 0 = pattern only.
# --n-pcs            Number of PCA components used to build the pattern (expression-similarity) space.
# --is-hexa          Flag for hex/Visium grids (doubles the effective spatial order). Omit for square grids.
# --is-padding       Flag to fill in missing grid spots by interpolating from neighbors.
# --filling-thres    Minimum fraction of occupied neighbors required to fill a missing spot (used with --is-padding).
# --output-layer     Name of the adata.layers entry that will store the smoothed matrix.
# --replace-x        Flag to also overwrite adata.X with the smoothed result (in addition to the output layer).
