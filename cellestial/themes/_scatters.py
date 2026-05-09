from lets_plot import (
    element_blank,
    element_text,
    scale_color_brewer,
    scale_size,
    theme,
    theme_classic,
)

_THEME_DIMENSION = (
    theme_classic()
    + theme(
        # customize all text
        text=element_text(color="#1f1f1f", family="Arial"),
        # customize all titles (includes legend)
        title=element_text(color="#1f1f1f", family="Arial"),
        # customize axis titles (labels)
        axis_title=element_text(color="#3f3f3f", family="Arial"),
        # customize legend text
        legend_text=element_text(color="#1f1f1f", size=11, face="plain"),
        # customize legend columns
    )
)

_THEME_SPATIAL = (
    theme_classic()
    + theme(
        text=element_text(color="#1f1f1f", family="Arial"),
        title=element_text(color="#1f1f1f", family="Arial"),
        axis_title=element_blank(),
        axis_text=element_blank(),
        axis_ticks=element_blank(),
        axis_line=element_blank(),
        legend_text=element_text(color="#1f1f1f", size=11, face="plain"),
    )
)

_THEME_SCATTER = theme(
    # customize all text
    text=element_text(color="#1f1f1f", family="Arial"),
    # customize all titles (includes legend)
    title=element_text(color="#1f1f1f", family="Arial"),
    # customize axis titles (labels)
    axis_title=element_text(color="#3f3f3f", family="Arial"),
    # customize legend text
    legend_text=element_text(color="#1f1f1f", size=11, face="plain"),
    # customize legend columns
) + scale_color_brewer(palette="Set2")

_THEME_SCATTER_BASE = theme(
    # customize all text
    text=element_text(color="#1f1f1f", family="Arial"),
    # customize all titles (includes legend)
    title=element_text(color="#1f1f1f", family="Arial"),
    # customize axis titles (labels)
    axis_title=element_text(color="#3f3f3f", family="Arial"),
    # customize legend text
    legend_text=element_text(color="#1f1f1f", size=11, face="plain"),
    # customize legend columns
)

_THEME_DOTPLOT = (
    theme_classic()
    + theme(
        panel_border=element_blank(),  # replaced by data-area geom_rect in dotplot
        axis_line=element_blank(),
        text=element_text(family="Arial", color="#1f1f1f"),
        axis_text_x=element_text(angle=90),
        axis_title=element_blank(),
        legend_text=element_text(size=10),
        legend_title=element_text(size=12),
        legend_box_spacing=0,
        legend_key_spacing_y=0,
    )
    #+ scale_y_discrete(expand=[0.05, 0.05])  # offsets for points from the frame
    #+ scale_x_discrete(expand=[0.025, 0.025])
    + scale_size(trans="sqrt", breaks=[25, 50, 75, 100])  # to be replaced by scale_size_area
)
