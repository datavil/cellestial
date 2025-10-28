from lets_plot import element_blank, element_text, ggsize, scale_fill_hue, theme, theme_classic

_THEME_DIST = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#1f1f1f"),
        title=element_text(family="Arial", color="#1f1f1f"),
        legend_title=element_text(family="Arial", color="#1f1f1f", face="Bold"),
        axis_title_x=element_blank(),
    )
    + scale_fill_hue()
    + ggsize(400, 400)
)

