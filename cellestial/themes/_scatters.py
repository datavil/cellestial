from lets_plot import (
    LetsPlot,
    aes,
    arrow,
    element_blank,
    element_line,
    element_text,
    geom_blank,
    geom_jitter,
    geom_point,
    geom_segment,
    geom_violin,
    gggrid,
    ggplot,
    ggsize,
    ggtb,
    guide_colorbar,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
    scale_color_brewer,
    scale_color_continuous,
    scale_color_gradient,
    scale_color_hue,
    scale_color_viridis,
    scale_shape,
    theme,
    theme_classic,
)

_THEME_DIMENSION = (
    theme_classic()
    + theme(

        # customize all text
        text=element_text(color="#1f1f1f", family="Arial", size=12, face="bold"),
        # customize all titles (includes legend)
        title=element_text(color="#1f1f1f", family="Arial"),
        # customize axis titles (labels)
        axis_title=element_text(color="#3f3f3f", family="Arial", size=18),
        # customize legend text
        legend_text=element_text(color="#1f1f1f", size=11, face="plain"),
        # customize legend columns

    )
    + ggsize(600, 400)
    + guides(color=guide_legend(ncol=2))
)


if __name__ == "__main__":
    pass
