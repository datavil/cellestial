from lets_plot import element_blank, element_text, theme

_THEME_HEATMAP = theme(
    panel_grid=element_blank(),
    axis_text_x=element_blank(),
    axis_title=element_blank(),
    # customize all text
    text=element_text(color="#1f1f1f", family="Arial"),
    # customize all titles (includes legend)
    title=element_text(color="#1f1f1f", family="Arial"),
    # customize legend text
    legend_text=element_text(color="#1f1f1f", size=11, face="plain"),
)
