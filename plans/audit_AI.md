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
| `_Container` | Claude Opus 5 |  |  |  |  |
| `_MuDataContainer` | Claude Opus 5 |  |  |  |  |
| `_container` | Claude Opus 5 |  |  |  |  |
| `_Container.observation_metadata` | Claude Opus 5 |  |  |  |  |
| `_Container.variable_metadata` | Claude Opus 5 |  |  |  |  |
| `_Container.observation_columns` | Claude Opus 5 |  |  |  |  |
| `_Container.variable_columns` | Claude Opus 5 |  |  |  |  |
| `_Container.observation_names` | Claude Opus 5 |  |  |  |  |
| `_Container.variable_names` | Claude Opus 5 |  |  |  |  |
| `_Container.n_observations` | Claude Opus 5 |  |  |  |  |
| `_Container.modality_names` | Claude Opus 5 |  |  |  |  |
| `_Container.select_modality` | Claude Opus 5 |  |  |  |  |
| `_Container.modality_column` | Claude Opus 5 |  |  |  |  |
| `_Container.observation_embeddings` | Claude Opus 5 |  |  |  |  |
| `_Container.variable_embeddings` | Claude Opus 5 |  |  |  |  |
| `_Container.owns_variable` | Claude Opus 5 |  |  |  |  |
| `_Container.resolve_variable` | Claude Opus 5 |  |  |  |  |
| `_Container.fetch_variable_columns` | Claude Opus 5 |  |  |  |  |
| `_MuDataContainer._without_modality_masks` | Claude Opus 5 |  |  |  |  |
| `_MuDataContainer._variable_column` | Claude Opus 5 |  |  |  |  |
| `_dendrogram_source` | Claude Opus 5 |  |  |  |  |
| `_read_dataset` | Claude Opus 5 |  |  |  |  |
| `pbmc_cite` | Claude Opus 5 |  |  |  |  |
| `AmbiguousVariableError` | Claude Opus 5 |  |  |  |  |
| `_Container.container_column` | Claude Opus 5 |  |  |  |  |
| `_container_column` | Claude Opus 5 |  |  |  |  |
| `_modality_source` | Claude Opus 5 |  |  |  |  |
| `_as_array` | Claude Opus 5 |  |  |  |  |
| `_qualified_alternatives` | Claude Opus 5 |  |  |  |  |
| `_Container._require_unique_variables` | Claude Opus 5 |  |  |  |  |
| `_resolve_pvalue_column` | Claude Opus 5 |  |  |  |  |

## Modified functions

Modified functions require only a line-by-line check. Repeated modifications are consolidated by function, with all sources retained.

| Function | Source | Line-by-line |
| --- | --- | --- |
| `_spatial_components` | Claude 4.7 |  |
| `spatial` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `spatials` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `cluster_outlines` | Mixed |  |
| `stream` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `arrow_axis` | Claude 4.7 |  |
| `ondata_legend` | Claude 4.7 |  |
| `dotplot` | Gemini 2.5, Claude 4.6, and Claude 4.7 and Claude Opus 5 |  |
| `heatmap` | Claude 4.6 and Claude 4.7 and Claude Opus 5 |  |
| `ridge` | Claude 4.7 and Claude Opus 5 |  |
| `ridges` | Claude 4.7 |  |
| `dimensional` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
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
| `anndata_observations_frame` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `anndata_variables_frame` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `build_frame` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `_resolve_tooltips` | Claude 4.7 |  |
| `xyplot` | Claude 4.7 and Claude 4.8 and Claude Opus 5 |  |
| `xyplots` | Claude 4.7 and Claude 4.8 |  |
| `expressions` | Claude 4.7 |  |
| `stacked_violin` | Claude 4.7 and Claude Opus 5 |  |
| `expression` | Claude 4.7 and Claude 4.8 |  |
| `volcanos` | Claude 4.8 and Claude Opus 5 |  |
| `highest_expressed_genes` | Claude 4.8 and Claude Opus 5 |  |
| `elbow` | Claude 4.8 and Claude Opus 5 |  |
| `anndata_variable_columns` | Claude Opus 5 |  |
| `_require_feature_key` | Claude Opus 5 |  |
| `_is_variable_key` | Claude Opus 5 |  |
| `_are_variables` | Claude Opus 5 |  |
| `_is_observation_key` | Claude Opus 5 |  |
| `_are_observations` | Claude Opus 5 |  |
| `_select_variable_keys` | Claude Opus 5 |  |
| `_collect_aes_columns` | Claude Opus 5 |  |
| `_is_observation_feature` | Claude Opus 5 |  |
| `_are_observation_features` | Claude Opus 5 |  |
| `_is_variable_feature` | Claude Opus 5 |  |
| `_are_variable_features` | Claude Opus 5 |  |
| `_resolve_embedding_key` | Claude Opus 5 |  |
| `_determine_axis` | Claude Opus 5 |  |
| `_get_dendrogram` | Claude Opus 5 |  |
| `from_url` | Claude Opus 5 |  |
| `annotated_heatmap` | Claude Opus 5 |  |
| `markers` | Claude Opus 5 |  |
| `volcano` | Claude Opus 5 |  |
| `marker_genes` | Claude Opus 5 |  |
| `marker_genes_dict` | Claude Opus 5 |  |
| `_build_volcano_frame` | Claude Opus 5 |  |
| `_range_inclusive` | Claude Opus 5 |  |
