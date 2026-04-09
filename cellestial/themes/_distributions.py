from lets_plot import (
    coord_flip,
    element_blank,
    element_text,
    scale_fill_hue,
    theme,
    theme_classic,
    ylab,
)

_THEME_DIST = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#1f1f1f"),
        title=element_text(family="Arial", color="#1f1f1f"),
        legend_title=element_text(family="Arial", color="#1f1f1f", face="Bold"),
        axis_title_x=element_blank(),
    )
    + scale_fill_hue()
)

_THEME_HIGHEST = (
    theme(
        text=element_text(family="Arial", color="#1f1f1f"),
        title=element_text(family="Arial", color="#1f1f1f"),
        legend_title=element_text(family="Arial", color="#1f1f1f", face="Bold"),
        axis_title_x=element_blank(),
        legend_position="none",
    )
    + coord_flip()
    + scale_fill_hue()
    + ylab("Percentage of Total Counts")
)
