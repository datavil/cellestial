from lets_plot import element_text, ggsize, scale_fill_viridis, theme, theme_classic

_THEME_VIOLIN = (
    theme_classic()
    + theme(
        text=element_text(family="Arial", color="#3f3f3f"),
        title=element_text(family="Arial", color="#3f3f3f"),
    )
    + scale_fill_viridis()
    + ggsize(400, 400)
)