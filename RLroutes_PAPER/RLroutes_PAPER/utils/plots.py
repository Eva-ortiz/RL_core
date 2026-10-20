from enum import StrEnum
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from RLroutes_PAPER.utils.drift import sec_2_day_hour_min


class Buoys_plot_text(StrEnum):
    """Selection of text to display in buoys plots.

    Attributes
    ----------
    VISIT_nd_KEY : str
        "VISIT_nd_KEY_text", order of visit and the buoy KEY of each point.
    KEY : str
        "KEY_text", buoy key of each point.
    DATETIME : str
        "datetime_text", datetime of the buoy at each location.
    """

    VISIT_nd_KEY = "VISIT_nd_KEY_text"
    KEY = "KEY_text"
    DATETIME = "datetime_text"


class RL_routes_plot:
    """Plot class to display buoys, part of a route or not."""

    def __init__(
        self,
        lat_col: str = "LATITUDE",
        lon_col: str = "LONGITUDE",
        reward_col: str = "REWARD",
        key_col: str = "KEY",
        date_col: str = "DATE",
        time_col: str = "TIME",
        save_path: Path | None = Path("./img/RL_route"),
    ):
        """Initialize fig, axis and additional attributes.

        Parameters
        ----------
        lat_col : str, optional
            Column name storing latitude, by default "LATITUDE".
        lon_col : str, optional
            Column name storing longitude, by default "LONGITUDE".
        reward_col : str, optional
            Column name storing reward, by default "REWARD".
        key_col : str, optional
            Column name storing buoy KEY, by default "KEY".
        date_col : str, optional
            Column name storing date, by default "DATE"
        time_col : str, optional
            Column name storing time, by default "TIME"
        save_path : Path | None, optional
            Route to save output images, filename included.
            By default, Path("./img/RL_route").
        """
        self.fig, self.ax = plt.subplots()
        plt.margins(0.1, 0.1)

        self.lat_col, self.lon_col = lat_col, lon_col
        self.reward_col, self.key_col = reward_col, key_col
        self.date_col, self.time_col = date_col, time_col
        self.save_path = save_path

    def plot_buoys(
        self,
        df_env_buoys: pd.DataFrame,
        df_visited_buoys: pd.DataFrame | None,
        text: Buoys_plot_text | None = None,
        more_info: dict[str, int | float] | None = None,
        title: str = "Reinforcement Learning",
        save: bool = True,
    ) -> tuple[Figure, Axes]:
        """Plot scatter and (if desired) route of input buoys.

        Parameters
        ----------
        df_env_buoys : pd.DataFrame
            Scattered buoys to plot, not forming part of the route, with
            `lat_col` and `lon_col`.
        df_visited_buoys : pd.DataFrame | None
            Buoys forming part of the route to plot, i.e., buoys data with
            `lat_col`, `lon_col` and `reward_col` associated; in addition to
            `key_col`, `date_col` and `time_col`, if desired.
        text : Buoys_plot_text | None, optional
            Select the text to display on each route lat/lon if
            `df_visited_buoys` is not None, else, on each scattered buoy of
            `df_env_buoys`.
            * VISIT_nd_KEY : order of visit and KEY of each point.
            * KEY : KEY of each point.
            * DATETIME : datetime of visit to each location.
            If None, do not display any additional text. By default, None.
        more_info: dict[str, int | float] | None, optional
            Info to add to the title or caption. Accepted fields are:
            * For title: `all_buoys`, `t_train`, `t_optimization`, `n_episodes`,
            `n_episodes_per_seed`
            * For caption: `n_buoys`, `capture`, `t_travel`, `t_stop`
            If None, do not add info to title or caption. By default, None.
        title : str, optional
            String with main info for title, by default "Reinforcement
            Learning".
        save : bool, optional
            Whether to save the resulting figure or not, by default True.

        Returns
        -------
        Figure
            Output figure with scatter plot.
        Axes
            Output axes with scatter plot.

        See Also
        --------
        _info_to_fig : more information about `more_info` input.
        """
        # plot all environment buoys
        self.ax.scatter(
            df_env_buoys[self.lon_col],
            df_env_buoys[self.lat_col],
            edgecolors="black",
            facecolors="None",
            linewidths=1,
        )
        # set x/y-labels
        self.ax.set_xlabel("Longitude (deg)")
        self.ax.set_ylabel("Latitude (deg)")

        # plot scatter of visited buoys and set its color based on retrieved reward
        if df_visited_buoys is not None:
            scatter = self.ax.scatter(
                df_visited_buoys[self.lon_col],
                df_visited_buoys[self.lat_col],
                c=df_visited_buoys[self.reward_col],
                cmap="viridis",
            )
            cbar = plt.colorbar(scatter, ax=self.ax)
            cbar.set_label("Reward (a.u.)")

        if text is not None:
            self._text_at_location(
                df_visited_buoys if df_visited_buoys is not None else df_env_buoys,
                text=text,
            )

        if more_info is not None:
            self._info_to_fig(more_info, suptitle_str=title)

        if save:
            self.fig.tight_layout()
            self.fig.savefig(
                self.save_path.with_name(
                    self.save_path.stem
                    + (f"_{text}.png" if text is not None else ".png")
                )
            )
        return self.fig, self.ax

    def _text_at_location(self, df_buoys: pd.DataFrame, text: Buoys_plot_text):
        """Set desired text for each location.

        Warnings
        --------
        * If VISIT_nd_KEY text, visit order is assumed to be df index.
        """
        lon_series = df_buoys[self.lon_col]
        lat_series = df_buoys[self.lat_col]

        # set an small displacement to have more legible annotations
        text_displace_lon = (lon_series.max() - lon_series.min()) * 0.075
        text_displace_lat = (lat_series.max() - lat_series.min()) * 0.025

        # print visit order and buoyKEY
        if text == Buoys_plot_text.VISIT_nd_KEY or text == Buoys_plot_text.KEY:
            try:
                KEY_series = df_buoys[self.key_col]
            except KeyError as err:
                raise KeyError(
                    "`key_col` necessary in data for adding `KEY_text`"
                ) from err

            for idx_visit, buoyKEY in KEY_series.items():
                if text == Buoys_plot_text.VISIT_nd_KEY and idx_visit < (
                    KEY_series.shape[0] - 1
                ):
                    lon_distance = lon_series[idx_visit + 1] - lon_series[idx_visit]
                    lat_distance = lat_series[idx_visit + 1] - lat_series[idx_visit]
                    # arrow indicating the transition
                    self.ax.annotate(
                        "",
                        xy=(
                            lon_series[idx_visit + 1] - 0.02 * lon_distance,
                            lat_series[idx_visit + 1] - 0.02 * lat_distance,
                        ),  # shorten the arrow to avoid overlapping
                        xytext=(
                            lon_series[idx_visit] + 0.02 * lon_distance,
                            lat_series[idx_visit] + 0.02 * lat_distance,
                        ),
                        arrowprops=dict(arrowstyle="->", color="grey", lw=1),
                    )
                    # annotate transition number of each arrow
                    self.ax.annotate(
                        idx_visit,
                        (
                            lon_series[idx_visit + 1]  # last lon
                            - (lon_distance / 2),  # lon to the center of the arrow
                            lat_series[idx_visit + 1]  # last lat
                            - (lat_distance / 2)  # lat to the center of the arrow
                            + text_displace_lat,  # displacement above the arrow
                        ),
                        color="blue",
                        fontweight="bold",
                    )
                # annotate buoy KEYs
                self.ax.annotate(
                    buoyKEY,
                    (
                        lon_series[idx_visit] - text_displace_lon,
                        lat_series[idx_visit] + text_displace_lat,
                    ),
                )

        # print date and time of each visit
        elif text == Buoys_plot_text.DATETIME:
            try:
                dates_series = df_buoys[self.date_col]
                time_series = df_buoys[self.time_col]
            except KeyError as err:
                raise KeyError(
                    "date and time cols necessary in data for adding `datetime_text`"
                ) from err

            for idx, date in dates_series.items():
                self.ax.annotate(
                    f"{time_series[idx].strftime('%H:%M')}, {date}",
                    (
                        lon_series[idx] - text_displace_lon,
                        lat_series[idx] + text_displace_lat,
                    ),
                )
        else:
            raise NotImplementedError("Selected text option not implemented.")

    def _info_to_fig(self, info: dict[str, int | float], suptitle_str: str = ""):
        """Add provided info to fig title/caption.

        Parameters
        ----------
        info : dict[str, int | float]
            Info to add to the title/caption. Accepted fields are:
            For title:
            * all_buoys: total number of buoys
            * t_train: training time (s)
            * t_optimization: hyperparameter optimization time (s)
            * n_episodes: number of explored routes (episodes)
            * n_episodes_per_seed: number of explored routes (episodes) per seed
            For caption:
            * n_buoys: number of buoys of the optimal route
            * capture: total capture of the optimal route (tons)
            * t_travel: travel time of the optimal route (days)
            * t_stop: total stop time of the optimal route (days)
        suptitle_str : str, optional
            String with main title, by default "".
        """
        # add input info
        expected_info = [
            "all_buoys",
            "t_train",
            "t_optimization",
            "n_episodes",
            "n_episodes_per_seed",
            "n_buoys",
            "capture",
            "t_travel",
            "t_stop",
        ]
        # check all input info is considered
        if not set(info.keys()) <= set(expected_info):
            raise ValueError(
                "`info` input keys not expected. Double check provided docstring."
            )

        title_substring = ""
        if "all_buoys" in info:
            title_substring += f" total buoys: {info['all_buoys']};"
        if "t_train" in info:
            train_days, train_hours, train_mins = sec_2_day_hour_min(info["t_train"])
            title_substring += (
                " training time: "
                + f"{train_days}:{train_hours}:{train_mins:.2f} (d:h:m);"
            )
        if "n_episodes" in info:
            title_substring += f"\n explored routes: {info['n_episodes']};"
        if "n_episodes_per_seed" in info:
            routes_per_seed = info["n_episodes_per_seed"]
            title_substring += (
                f"\n explored routes ({len(routes_per_seed)} seeds): "
                f"{mean(routes_per_seed):.2f} mean, {sum(routes_per_seed)} total;"
            )
        if "t_optimization" in info:
            opt_days, opt_hours, opt_mins = sec_2_day_hour_min(info["t_optimization"])
            title_substring += (
                " hp optimization time: "
                + f"{opt_days}:{opt_hours}:{opt_mins:.2f} (d:h:m);"
            )

        caption_substring = ""
        if "n_buoys" in info:
            caption_substring += f" n buoys: {info['n_buoys']};"
        if "capture" in info:
            caption_substring += f" total capture: {info['capture']:.2f} (T);"
        if "t_travel" in info:
            caption_substring += f"\n travel time: {info['t_travel']:.2f} (days);"
        if "t_stop" in info:
            caption_substring += f" stop time: {info['t_stop']:.2f} (days);"

        # set title and caption
        self.fig.suptitle(
            suptitle_str + f"\n({title_substring})" if title_substring else ""
        )
        if caption_substring:
            self.fig.supxlabel("Best route\n" + caption_substring)

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
