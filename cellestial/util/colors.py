from lets_plot import *

LetsPlot.setup_html()

# Hand-picked colors for cellestial
TEAL = "#219B9D"
RED = "#D2042D"
CHERRY = "#AF1740"
BLUE = "#377EB8"
LIGHT_GRAY = "#E6E6E6"


def show_colors():
    colors = {"TEAL": TEAL, "RED": RED, "CHERRY": CHERRY, "BLUE": BLUE, "LIGHT_GRAY": LIGHT_GRAY}

    plots = []
    for color in colors:
        text_color = "white" if color != "LIGHT_GRAY" else "black"

        plot = (
            ggplot({"x": [1], "y": [1]})
            + geom_point(aes(x="x", y="y"), fill=colors[color], shape=22, size=90, color="#1f1f1f")
            + geom_text(
                aes(x="x", y="y"),
                label=f"{color}\n",
                color=text_color,
                size=14,
                fontface="bold",
                family="Arial",
            )
            + geom_text(
                aes(x="x", y="y"),
                label=f"\n{colors[color]}",
                color=text_color,
                size=10,
                fontface="regular",
                family="sans-serif",
            )
            + theme_void()
        )
        plots.append(plot)

    ncol = 3
    nrow = len(colors) // ncol
    if len(colors) % ncol != 0:
        nrow += 1

    return gggrid(plots, ncol=3, hspace=0, vspace=0) + ggsize(ncol * 200, nrow * 200)


if __name__ == "__main__":
    show_colors().to_html("colors.html")