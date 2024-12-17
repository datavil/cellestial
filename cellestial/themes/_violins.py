from lets_plot import element_text, ggsize, scale_fill_hue, theme, theme_classic

_THEME_VIOLIN = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#3f3f3f"),
        title=element_text(family="Arial", color="#3f3f3f"),
        legend_title=element_text(family="Arial", color="#3f3f3f", face="Bold"),
    )
    + scale_fill_hue()
    + ggsize(400, 400)
)

_THEME_BOXPLOT = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#3f3f3f"),
        title=element_text(family="Arial", color="#3f3f3f"),
        legend_title=element_text(family="Arial", color="#3f3f3f", face="Bold"),
    )
    + scale_fill_hue()
    + ggsize(400, 400)
)
