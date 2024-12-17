from lets_plot import element_text, ggsize, scale_fill_hue, theme, theme_classic

_THEME_VIOLIN = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#1f1f1f"),
        title=element_text(family="Arial", color="#1f1f1f"),
        legend_title=element_text(family="Arial", color="#1f1f1f", face="Bold"),
    )
    + scale_fill_hue()
    + ggsize(400, 400)
)

_THEME_BOXPLOT = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#1f1f1f"),
        title=element_text(family="Arial", color="#1f1f1f"),
        legend_title=element_text(family="Arial", color="#1f1f1f", face="Bold"),
    )
    + scale_fill_hue()
    + ggsize(400, 400)
)
