# AI Audit

AI-generated and AI-modified functions are tracked separately below. A blank check cell means the review is still outstanding.

## Generated functions

Generated functions require behavioral, visual, edge-case, and line-by-line checks where applicable.

| Function | Source | Behavioral | Visual | Edge cases | Line-by-line |
| --- | --- | --- | --- | --- | --- |
| `_THEME_SPATIAL` | Claude 4.7 | ✓ | ✓ | ✓ | ✓ |
| `DeferredLayer` | Claude 4.7 | ✓ | ✓ |  |  |
| `bracket` | Claude 4.7 | ✓ | ✓ |  |  |
| `matrixplot` | Claude 4.7 | ✓ | ✓ |  |  |
| `stacked_violin` | Claude 4.7 | ✓ | ✓ |  |  |
| `_bin_within_groups` | Claude 4.7 |  |  |  |  |
| `volcano` | Claude 4.7 | ✓ | ✓ |  |  |
| `volcanos` | Claude 4.7 | ✓ | ✓ |  |  |
| `_build_volcano_frame` | Claude 4.7 | ✓ | ✓ |  |  |
| `markers` | Claude 4.7 | ✓ | ✓ |  |  |
| `_build_markers_frame` | Claude 4.7 | ✓ | ✓ |  |  |
| `marker_genes` | Claude 4.7 | ✓ | ✓ |  |  |
| `marker_genes_dict` | Claude 4.7 | ✓ | ✓ |  |  |
| `_marker_names_per_group` | Claude 4.7 | ✓ | ✓ |  |  |
| `_get_dendrogram` | Claude 4.6 | ✓ | ✓ | ✓ |  |
| `_get_dendrogram_path_frame` | Claude 4.6 | ✓ | ✓ |  |  |
| `_select_embedding_keys` | Claude 4.7 |  |  |  |  |
| `_collect_aes_columns` | Claude 4.7 |  |  |  |  |
| `_resolve_instance_key` | Claude 4.7 |  |  |  |  |
| `layout` | Claude 4.8 |  |  |  |  |
| `_normalize_widths` | Claude 4.8 |  |  |  |  |
| `_unsupported_data_type` | Claude 4.8 |  |  |  |  |
| `_reject_sequence_key` | Claude 4.8 |  |  |  |  |
| `_require_feature_key` | Claude 4.8 |  |  |  |  |
| `histogram` | Claude 4.8 |  |  |  |  |
| `histograms` | Claude 4.8 |  |  |  |  |
| `annotated_heatmap` | Claude 4.8 |  |  |  |  |
| `_order_observations` | Claude 4.8 |  |  |  |  |
| `_annotation_strip` | Claude 4.8 |  |  |  |  |
| `_as_list` | Claude 4.8 |  |  |  |  |
| `_bin_observations` | Claude 4.8 |  |  |  |  |
| `_assign_position_y` | Claude 4.8 |  |  |  |  |
| `_annotation_scale` | Claude 4.8 |  |  |  |  |
| `_dendrogram_panel` | Claude 4.8 |  |  |  |  |
| `_blank_strip_theme` | Claude 4.8 |  |  |  |  |
| `_as_layers` | Claude 4.8 |  |  |  |  |
| `_apply_layers` | Claude 4.8 |  |  |  |  |
| `_validate_aesthetic_columns` | Claude Opus 5 |  |  |  |  |

## Modified functions

Modified functions require only a line-by-line check. Repeated modifications are consolidated by function, with all sources retained.

| Function | Source | Line-by-line |
| --- | --- | --- |
| `_spatial_components` | Claude 4.7 |  |
| `spatial` | Claude 4.7 and Claude 4.8 |  |
| `spatials` | Claude 4.7 and Claude 4.8 |  |
| `cluster_outlines` | Mixed |  |
| `stream` | Claude 4.7 and Claude 4.8 |  |
| `arrow_axis` | Claude 4.7 |  |
| `ondata_legend` | Claude 4.7 |  |
| `dotplot` | Gemini 2.5, Claude 4.6, and Claude 4.7 |  |
| `heatmap` | Claude 4.6 and Claude 4.7 |  |
| `ridge` | Claude 4.7 |  |
| `ridges` | Claude 4.7 |  |
| `dimensional` | Claude 4.7 and Claude 4.8 |  |
| `umap` | Claude 4.7 |  |
| `tsne` | Claude 4.7 |  |
| `pca` | Claude 4.7 |  |
| `dimensionals` | Claude 4.7 |  |
| `umaps` | Claude 4.7 |  |
| `tsnes` | Claude 4.7 |  |
| `pcas` | Claude 4.7 |  |
| `violin` | Claude 4.7 |  |
| `boxplot` | Claude 4.7 |  |
| `violins` | Claude 4.7 |  |
| `boxplots` | Claude 4.7 |  |
| `_distribution` | Claude 4.7, Claude 4.8, and Claude Opus 5 |  |
| `anndata_observations_frame` | Claude 4.7 and Claude 4.8 |  |
| `anndata_variables_frame` | Claude 4.7 and Claude 4.8 |  |
| `build_frame` | Claude 4.7 and Claude 4.8 |  |
| `_resolve_tooltips` | Claude 4.7 |  |
| `xyplot` | Claude 4.7 and Claude 4.8 |  |
| `xyplots` | Claude 4.7 and Claude 4.8 |  |
| `expressions` | Claude 4.7 |  |
| `stacked_violin` | Claude 4.7 |  |
| `expression` | Claude 4.7 and Claude 4.8 |  |
| `volcanos` | Claude 4.8 |  |
| `highest_expressed_genes` | Claude 4.8 |  |
| `elbow` | Claude 4.8 |  |
