import polars as pl
from lets_plot import *
from lets_plot import (
    LetsPlot,
    arrow,
    element_blank,
    geom_segment,
    ggtb,
    labs,
    theme,
)

LetsPlot.setup_html()


# ------------------------------ EXAMPLE STRUCTURE ------------------------------
def example(func):
    def wrapper(*args, **kwargs):
        # MUST DO: merge the default kwargs with the user-provided kwargs
        all_kwargs = func.__kwdefaults__
        all_kwargs.update(kwargs)
        # ------------------------------------------------------

        """ modify the output
        result = func(*args, **kwargs)

        # handle the case
        if all_kwargs.get("example"):
            result += something 
        else:
            pass
            
        return result
        """

    # MUST DO: inherit the default kwargs
    wrapper.__kwdefaults__ = func.__kwdefaults__  # inherit the default kwargs
    return wrapper


# ------------------------------ INTERACTIVE ------------------------------
def interactive(func):
    def wrapper(*args, **kwargs):
        # merge the default kwargs with the user-provided kwargs
        all_kwargs = func.__kwdefaults__
        all_kwargs.update(kwargs)
        # ------------------------------------------------------

        # get the value of the `interactive` kwarg
        inter = all_kwargs.get("interactive")
        if inter is True:
            return func(*args, **kwargs) + ggtb()
        elif inter is False:
            return func(*args, **kwargs)
        else:
            msg = f"expected True or False for 'interactive' argument, but received {inter}"
            raise ValueError(msg)

    wrapper.__kwdefaults__ = func.__kwdefaults__  # inherit the default kwargs
    return wrapper


def arrow_axis(func):
    def modifier(*args, **kwargs):
        # merge the default kwargs with the user-provided kwargs
        all_kwargs = func.__kwdefaults__
        all_kwargs.update(kwargs)
        # ------------------------------------------------------

        # base plot
        plot = func(*args, **kwargs)

        # get arguments
        data = args[0]
        axis_type = all_kwargs.get("axis_type")
        arrow_size = all_kwargs.get("arrow_size")
        arrow_color = all_kwargs.get("arrow_color")
        arrow_size = all_kwargs.get("arrow_size")
        arrow_angle = all_kwargs.get("arrow_angle")
        arrow_length = all_kwargs.get("arrow_length")
        dimensions = all_kwargs.get("dimensions")

        if axis_type is None:
            plot += theme(
                # remove axis elements
                axis_text_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks_y=element_blank(),
                axis_ticks_x=element_blank(),
                axis_line=element_blank(),
            )

        elif axis_type == "arrow":
            frame = pl.from_numpy(
                data.obsm[f"X_{dimensions}"][:, :2], schema=[f"{dimensions}1", f"{dimensions}2"]
            )

            plot += theme(
                # remove axis elements
                axis_text_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks_y=element_blank(),
                axis_ticks_x=element_blank(),
                axis_line=element_blank(),
                # # position axis titles according to arrow size
                # axis_title_x=element_text(color="#3f3f3f", family="Arial", size=18,hjust=arrow_size / 2),
                # axis_title_y=element_text(color="#3f3f3f", family="Arial", size=18,hjust=arrow_size / 2),
            )
            x_max = frame.select(f"{dimensions}1").max().item()
            x_min = frame.select(f"{dimensions}1").min().item()
            y_max = frame.select(f"{dimensions}2").max().item()
            y_min = frame.select(f"{dimensions}2").min().item()

            # find total difference between the max and min for both axis
            x_diff = x_max - x_min
            y_diff = y_max - y_min

            # find the ends of the arrows
            xend = x_min + arrow_length * x_diff
            yend = y_min + arrow_length * y_diff

            # adjust bottom ends of arrows
            adjust_rate = 0.05
            x0 = x_min - x_diff * adjust_rate
            y0 = y_min - y_diff * adjust_rate

            # X axis
            plot += geom_segment(
                x=x0,
                y=y0,
                xend=xend,
                yend=y0,
                color=arrow_color,
                size=arrow_size,
                arrow=arrow(arrow_angle),
            )
            # Y axis
            plot += geom_segment(
                x=x0,
                y=y0,
                xend=x0,
                yend=yend,
                color=arrow_color,
                size=arrow_size,
                arrow=arrow(arrow_angle),
            )

            plot += labs(
                x = f"{dimensions}1".upper(),
                y = f"{dimensions}2".upper(),
            )


        elif axis_type == "axis":
            pass

        return plot

    modifier.__kwdefaults__ = func.__kwdefaults__  # inherit the default kwargs
    return modifier
