from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from classes.MyModel import MySine
from modules.config import GRAPH_DIR, PLOT_TITLE

SPACE = ' '
CMAP = 'Set3'
CMAP_HEATMAP = 'cividis'


def parameter_graph(
    filename: str,
    title: str,
    index_list: list,
    gradient_list: list,
    legend: list,
):
    """Plots a graph of the parameter evolution by iteration."""
    p = plt.plot(index_list, gradient_list)
    plt.grid(axis='x')
    plt.legend(handles=p, labels=legend, ncols=3, fontsize='xx-small', shadow='true')
    plt.title(title)
    plt.tight_layout()
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value in radians')
    plt.savefig(filename)
    plt.show()


def find_size(cost_list: list) -> tuple:
    """Find the number of subplots"""
    if cost_list:
        length = len(cost_list)
    else:
        length = 1
    if length == 1:
        rows = 1
        columns = 1
    elif length % 2 != 0:
        rows = length
        columns = 1
    else:
        rows = int(length / 2)
        columns = 2
    return (length, rows, columns)


def find_graph_statistics(av_list: np.ndarray, best: float) -> tuple:
    """Helper function for graph plotting to find good values for ymin and ymax"""
    maximum = float(np.max(av_list))
    ymin, ymax = int(best * 0.9), int(maximum * 1.1)
    return (ymin, ymax)


def find_best_coords(x_list: np.ndarray, best: float) -> tuple:
    """Helper function for graph plotting to find coordinates for best known value line"""
    x1 = [0, x_list[-1]]
    y1 = [best, best]
    return (x1, y1)


def find_i_j(count: int, rows: int) -> tuple:
    """Find indices for subplots"""
    if count < rows:
        i, j = count, 0
    else:
        i, j = count - rows, 1
    return (i, j)


def cost_graph_multi(
    filename: str,
    parameter_list: list = None,
    x_list: list = None,
    av_list: list = None,
    lowest_list: list = None,
    sliced_list: list = None,
    best: float = None,
    main_title: str = '',
    sub_title: str = '',
    x_label: str = 'Epoch',
    figsize: tuple = (8, 8),
):
    """Plots a graph of the cost function for multiple lists"""
    plt.style.use('seaborn-v0_8-colorblind')
    length, rows, columns = find_size(parameter_list)
    fig, axs = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    fig.suptitle(main_title)
    ymin, ymax = find_graph_statistics(av_list, best)
    x1, y1 = find_best_coords(x_list, best)

    for count in range(length):
        i, j = find_i_j(count, rows)
        axs[i, j].plot(
            x_list, av_list[count], linewidth=1.0, color='blue', label='Average'
        )
        if sliced_list:
            axs[i, j].plot(
                x_list,
                sliced_list[count],
                linewidth=1.0,
                color='orange',
                label='Sliced Average',
            )
        axs[i, j].plot(
            x_list, lowest_list[count], linewidth=1.0, color='red', label='Lowest found'
        )
        axs[i, j].plot(x1, y1, linewidth=1.0, color='black', label='Lowest known')
        axs[i, j].grid(axis='x', color='0.95')
        axs[i, j].axis(ymin=ymin, ymax=ymax)
        axs[i, j].set_xlabel(x_label, fontsize=6)
        axs[i, j].set_ylabel('Energy (Distance)', fontsize=6)
        axs[i, j].xaxis.set_tick_params(labelsize=7)
        axs[i, j].yaxis.set_tick_params(labelsize=7)
        if sub_title != '':
            sub_title_full = sub_title + f'{parameter_list[count]}'
            axs[i, j].set_title(sub_title_full, fontsize=6)
        axs[i, j].legend(fontsize=6, loc='upper right', shadow='true')
    fig.tight_layout()
    fig.savefig(filename)
    # plt.close(fig)


def plot_shortest_routes(points: list, route1: list, route2: list = None):
    """Plot the shortest route found and optionally a hot start route."""
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, marker='o')
    for i, point_index in enumerate(route1):
        plt.annotate(str(point_index), (x[point_index], y[point_index]))
    plt.title('Route through locations selected at random')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)

    if route2:
        for i in range(len(route2) - 1):
            start_index = route2[i]
            end_index = route2[i + 1]
            plt.plot(
                [x[start_index], x[end_index]],
                [y[start_index], y[end_index]],
                color='red',
            )
        # Connect the last point back to the first to complete the cycle
        start_index = route2[-1]
        end_index = route2[0]
        plt.plot(
            [x[start_index], x[end_index]], [y[start_index], y[end_index]], color='red'
        )

    for i in range(len(route1) - 1):
        start_index = route1[i]
        end_index = route1[i + 1]
        plt.plot(
            [x[start_index], x[end_index]], [y[start_index], y[end_index]], color='blue'
        )
        # Connect the last point back to the first to complete the cycle
        start_index = route1[-1]
        end_index = route1[0]
        plt.plot(
            [x[start_index], x[end_index]], [y[start_index], y[end_index]], color='blue'
        )

    red_patch = mpatches.Patch(color='red', label='The hot start route')
    blue_patch = mpatches.Patch(color='blue', label='The shortest route')
    plt.legend(handles=[red_patch, blue_patch], shadow='true')
    plt.show()
    plt.close()


def plot_sine_activation():
    """Plot the Sine Activation Function for the classical ML model."""
    title = 'Sine_Activation_Function'
    filepath = Path(GRAPH_DIR).joinpath(title + '.pdf')
    # create custom dataset
    x = torch.linspace(0, 1, 100)
    k = MySine()
    y = k(x)
    # Plot the Sine Activation Function
    plt.plot(x, y)
    plt.grid(True)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_3d_graph_models(
    grouped_means: pd.DataFrame, input: str, input2: str = 'layers'
):
    """Plot a 3D bar graph of the given input data grouped by layers and locations

    Parameters
    ----------
    grouped_means : pd.DataFrame
        DataFrame containing the grouped means data.
    input : str
        The column name for the z-axis values.
    input2 : str, optional
        The column name for the y-axis values (default is 'layers').
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    input2_vals = sorted(grouped_means[input2].unique())
    locations = sorted(grouped_means['locations'].unique(), reverse=True)

    input2_map = {sli: i for i, sli in enumerate(input2_vals)}
    loc_map = {loc: i for i, loc in enumerate(locations)}

    # Assign colors for each location
    colors = plt.get_cmap(CMAP, len(input2_vals))  # or 'Set3', 'Paired', etc.
    input2_colors = {item: colors(i) for i, item in enumerate(input2_vals)}

    # Bar sizes
    dx = 0.5
    dy = 0.15

    # Plot bars with different colors
    for i, row in grouped_means.iterrows():
        x = loc_map[row['locations']] - dx / 2  # Center the bar on the x-axis
        y = input2_map[row[input2]] - dy / 2  # Center the bar on the y-axis
        z = 0
        dz = row[input]

        color = input2_colors[row[input2]]
        ax.bar3d(x, y, z, dx, dy, dz, color=color, shade=True)

    # Label axes
    ax.set_xlabel('Locations')
    ax.set_ylabel(input2)
    ax.set_zlabel(input)

    # Set tick labels
    ax.set_xticks(list(loc_map.values()))
    ax.set_xticklabels(list(loc_map.keys()))
    ax.set_yticks(list(input2_map.values()))
    ax.set_yticklabels(list(input2_map.keys()))

    legend_handles = [
        mpatches.Patch(color=input2_colors[layer], label=layer) for layer in input2_vals
    ]
    plt.legend(
        handles=legend_handles,
        title=input2,
        loc='upper left',
        bbox_to_anchor=(1, 1),
        shadow='true',
    )
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    formatted_input = input.replace('_', SPACE).lower()
    title = f'3D bar graph of {formatted_input} by {input2} and locations'
    if PLOT_TITLE:
        plt.title(title)
    filepath = Path(GRAPH_DIR).joinpath(f'{title}.pdf')

    plt.savefig(filepath, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_3d_graph_slice(
    grouped_means: pd.DataFrame, input: str, show_sem: bool = False
):
    """Plot a 3D bar graph of the given input data grouped by locations and slice.

    Parameters
    ----------
    grouped_means : pd.DataFrame
        DataFrame containing the grouped means data.
    input : str
        The column name for the z-axis values.
    show_sem : bool, optional
        Whether to show standard error of the mean (SEM) as error bars (default is False
    """

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Map categorical data to numeric positions
    locations = sorted(grouped_means['locations'].unique())
    slices = sorted(grouped_means['slice'].unique())
    loc_map = {loc: i for i, loc in enumerate(locations)}
    slice_map = {sli: i for i, sli in enumerate(slices)}

    # Assign colors for each location
    colors = plt.get_cmap(CMAP, len(locations))  # or 'Set3', 'Paired', etc.
    location_colors = {loc: colors(i) for i, loc in enumerate(locations)}

    # Bar and cap width sizes
    dx = 0.5
    dy = 0.25
    cap_width = 0.1

    # loop BY LOCATION ORDER so that the bars for higher locations are drawn on top of the bars for lower locations, making them more visible.
    for loc in locations:  # so 15 is drawn last
        df_loc = grouped_means[grouped_means['locations'] == loc]
        for _, row in df_loc.iterrows():
            x = slice_map[row['slice']] - dx / 2
            y = loc_map[loc] - dy / 2
            dz = row[input]

            ax.bar3d(
                x,
                y,
                0,
                dx,
                dy,
                dz,
                color=location_colors[loc],
                shade=True,
                zsort='max',  # puts smaller bars in front of taller bars, making them more visible
            )

            if show_sem and row['sem'] > 0:
                x_center = x + dx / 2
                y_center = y + dy / 2

                ax.plot(
                    [x_center, x_center],
                    [y_center, y_center],
                    [dz, dz + row['sem']],
                    color='black',
                    linewidth=1.5,
                )

                ax.plot(
                    [x_center - cap_width, x_center + cap_width],
                    [y_center, y_center],
                    [dz + row['sem'], dz + row['sem']],
                    color='black',
                    linewidth=1.5,
                )

    # View so Y is true depth axis
    ax.view_init(elev=20, azim=-60)

    # Label axes
    ax.set_xlabel('Slice')
    ax.set_ylabel('Locations')
    ax.set_zlabel(input)

    # Set tick labels
    ax.set_xticks(list(slice_map.values()))
    ax.set_xticklabels(list(slice_map.keys()))
    ax.set_yticks(list(loc_map.values()))
    ax.set_yticklabels(list(loc_map.keys()))

    legend_handles = [
        mpatches.Patch(color=location_colors[loc], label=loc) for loc in locations
    ]
    plt.legend(
        handles=legend_handles,
        title='Locations',
        loc='upper left',
        bbox_to_anchor=(1, 1),
        shadow='true',
    )

    formatted_input = input.replace('_', SPACE).lower()
    title = f'3D bar graph of {formatted_input} by location and slice'
    if PLOT_TITLE:
        plt.title(title)
    filepath = Path(GRAPH_DIR).joinpath(f'{title}.pdf')

    plt.savefig(filepath, bbox_inches='tight')

    plt.show()
    plt.close(fig)


def plot_heatmap(
    input: pd.DataFrame,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot a heat map of the given input data."""
    # Prepare the data
    heatmap_data = input.values
    x_labels = input.columns.astype(str)
    y_labels = input.index

    # Create the plot
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap_data, aspect='auto', cmap=CMAP_HEATMAP)

    # Add colorbar
    plt.colorbar(im, label='Solution Quality')

    # Set axis ticks and labels
    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)

    # Annotate each cell with the numeric value
    for i in range(heatmap_data.shape[0]):  # rows
        for j in range(heatmap_data.shape[1]):  # columns
            value = heatmap_data[i, j]
            if not np.isnan(value):  # skip missing values
                if value > 80:
                    color = 'black'
                else:
                    color = 'white'
                plt.text(j, i, f'{value:.1f}', ha='center', va='center', color=color)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_2d_graph_slice(
    sliced_summary, input='error', location_value=None, show_sem=True
):
    """
    Plot 2D means with optional SEM error bars for one location across slices.

    Parameters
    ----------
    sliced_summary : pd.DataFrame
        DataFrame with columns:
        ['locations', 'slice', input, 'sem']
    input : str, default='error'
        Column name of the metric to plot
    location_value : str or int
        Location value to plot (required)
    show_sem : bool, default=True
        Whether to show SEM error bars
    """

    if location_value is None:
        raise ValueError('location_value must be specified')

    # Filter data for the chosen location
    df_loc = sliced_summary[sliced_summary['locations'] == location_value].sort_values(
        'slice'
    )

    slices = df_loc['slice']
    means = df_loc[input]
    sems = df_loc['sem']

    plt.figure(figsize=(8, 5))

    if show_sem:
        plt.errorbar(
            slices,
            means,
            yerr=sems,
            fmt='o-',
            capsize=4,
            label=f'Location {location_value}',
        )
    else:
        plt.plot(slices, means, marker='o', label=f'Location {location_value}')

    plt.xlabel('Slice')
    plt.ylabel(input)
    plt.title(f'{input.capitalize()} across slices\n(Location = {location_value})')
    plt.legend(shadow='true')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_optimiser_performance1(
    df_stats: pd.DataFrame,
    best_dist: pd.array,
    locations: int,
    shots: int,
    plot_last_av: bool,
    plot_best_found: bool,
    plot_best_dist: bool,
):
    """Plots a graph of the performance of optimisers for different sinam and iterations"""
    sigmas = df_stats['sigma'].unique()
    gradient_types = df_stats['gradient_type'].unique()
    colors = plt.cm.tab10(range(len(gradient_types)))
    markers = ['o', 'x', 's', '^', 'D', 'v', 'P', '*', 'X']

    fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()
    title = f'Optimser Performance by Sigma with Estimated Uncertainty for {locations} Locations and {shots} Shots'

    fig.suptitle(title, fontsize=16)

    x_ticks = [10, 50, 250, 1_250]
    x_tick_labels = ['10', '50', '250', '1,250']

    for ax, sigma in zip(axes, sigmas):
        subset_sigma = df_stats[df_stats['sigma'] == sigma]

        for color, gt, marker in zip(colors, gradient_types, markers):
            subset = subset_sigma[subset_sigma['gradient_type'] == gt]
            subset = subset.sort_values('iterations')
            if subset.empty:
                continue
            x = subset['iterations']

            if plot_last_av:
                y_last_av = subset['last_av_mean']
                y_last_av_std = subset['last_av_sem']
                ax.plot(
                    x,
                    y_last_av,
                    marker=marker,
                    color=color,
                    label=f'{gt} - Last average',
                )
                ax.fill_between(
                    x,
                    y_last_av - y_last_av_std,
                    y_last_av + y_last_av_std,
                    color=color,
                    alpha=0.2,
                )
            if plot_best_found:
                y_best_found = subset['best_found_mean']
                y_best_found_std = subset['best_found_sem']
                ax.plot(
                    x,
                    y_best_found,
                    marker=marker,
                    color=color,
                    label=f'{gt} - Best Found',
                )
                ax.fill_between(
                    x,
                    y_best_found - y_best_found_std,
                    y_best_found + y_best_found_std,
                    color=color,
                    alpha=0.2,
                )
        if plot_best_dist:
            ax.axhline(
                y=best_dist,
                color='black',
                linewidth=2,
                label='Lowest known distance' if ax is axes[0] else None,
            )

        # Titles and formatting per subplot
        ax.set_title(f'Sigma = {sigma}')
        ax.set_xscale('log')
        ax.set_xticks(x_ticks, labels=x_tick_labels)
        ax.grid(True)

    # Shared labels
    fig.supxlabel('Iterations (budget): log scale', fontsize=14)
    fig.supylabel('Energy (distance)', fontsize=14)

    # ONE legend (global)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center right',
        bbox_to_anchor=(1.07, 0.7),
        fontsize=14,
        shadow='true',
    )

    # Layout spacing
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filepath = Path(GRAPH_DIR).joinpath(f'{title}.pdf')
    plt.savefig(fname=filepath, bbox_inches='tight')
    plt.show()


def plot_optimiser_performance2(
    df_stats: pd.DataFrame,
    best_dist: pd.array,
    locations: int,
    shots: int,
    plot_last_av: bool,
    plot_best_found: bool,
    plot_best_dist: bool,
):
    """plot optimiser peformance for different optimisers and iterations by sigma"""
    sigmas = df_stats['sigma'].unique()
    gradient_types = df_stats['gradient_type'].unique()
    colors = plt.cm.tab10(range(len(gradient_types)))
    iterations = df_stats['iterations'].unique()
    markers = ['o', 'x', 's', '^', 'D', 'v', 'P', '*', 'X']

    fig, axes = plt.subplots(3, 3, figsize=(16, 16), sharex=True, sharey=True)
    axes = axes.flatten()
    title = f'Optimser Performance by Sigma for different budgets with Estimated Uncertainty for {locations} Locations and {shots} Shots'

    fig.suptitle(title, fontsize=16)

    x_ticks = sigmas
    x_tick_labels = [f'{sigma:.1f}' for sigma in sigmas]

    for ax, gradient_type in zip(axes, gradient_types):
        subset_gt = df_stats[df_stats['gradient_type'] == gradient_type]

        for color, iteration, marker in zip(colors, iterations, markers):
            subset = subset_gt[subset_gt['iterations'] == iteration]
            subset = subset.sort_values('iterations')
            if subset.empty:
                continue
            x = subset['sigma']

            if plot_last_av:
                y_last_av = subset['last_av_mean']
                y_last_av_std = subset['last_av_sem']
                ax.plot(
                    x,
                    y_last_av,
                    marker=marker,
                    color=color,
                    label=f'{iteration:.0f} - Last average',
                )
                ax.fill_between(
                    x,
                    y_last_av - y_last_av_std,
                    y_last_av + y_last_av_std,
                    color=color,
                    alpha=0.2,
                )
            if plot_best_found:
                y_best_found = subset['best_found_mean']
                y_best_found_std = subset['best_found_sem']
                ax.plot(
                    x,
                    y_best_found,
                    marker=marker,
                    color=color,
                    label=f'{iteration:.0f} - Best Found',
                )
                ax.fill_between(
                    x,
                    y_best_found - y_best_found_std,
                    y_best_found + y_best_found_std,
                    color=color,
                    alpha=0.2,
                )
        if plot_best_dist:
            ax.axhline(
                y=best_dist,
                color='black',
                linewidth=2,
                label='Lowest known distance' if ax is axes[0] else None,
            )

        # Titles and formatting per subplot
        ax.set_title(f'Optimiser = {gradient_type}')
        ax.set_xticks(x_ticks, labels=x_tick_labels)
        ax.grid(True)

    # Shared labels
    fig.supxlabel('Sigma', fontsize=14)
    fig.supylabel('Energy (distance)', fontsize=14)

    # ONE legend (global)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='center right',
        bbox_to_anchor=(1.07, 0.9),
        fontsize=14,
        shadow='true',
    )

    # Layout spacing
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filepath = Path(GRAPH_DIR).joinpath(f'{title}.pdf')
    plt.savefig(fname=filepath, bbox_inches='tight')
    plt.show()


def plot_overall_results(
    width: float,  # the width of the bars
    multiplier: float,  # controls magnitude of offset
    simulation_means: pd.DataFrame,
    simulation_errors: pd.DataFrame,
    colors: list,  # list of colours to used for the bars
    locs: list,  # locations to be plotted
    title: str = 'Solution Quality by Number of Locations for VQA, ML, Monte Carlo and Greedy methods',
    greedy_classical: pd.DataFrame = False,
    AWS_results: pd.DataFrame = False,
    AWS_color: str = '#9467bd',
    AWS_xytext: tuple = (20, 5),
    monte_carlo: pd.DataFrame = False,
    n_cols: int = 2,  # number of columns in the legend
    legend_fontsize: str = 'small',  # font size for the bar labels
):
    """plots overall results for the paper"""
    x = np.arange(len(locs))
    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in simulation_means.items():
        offset = width * multiplier
        errors = simulation_errors[attribute]
        color = colors[attribute]
        rects = ax.bar(
            x=x + offset,
            height=measurement,
            width=width,
            label=attribute,
            yerr=errors,
            capsize=4,
            error_kw={'elinewidth': 1, 'alpha': 0.9},
            color=color,
            edgecolor='black',
            linewidth=0.6,
        )

        ax.bar_label(
            container=rects,
            padding=5,
            fmt='%.1f',
            label_type='edge',
            fontsize=8,
            rotation=90,
        )
        multiplier += 1

    num_bar_groups = 4  # total plotted groups including spacing
    group_width = width * (num_bar_groups - 1)
    center_offset = group_width / 2 - width / 2  # center alignment

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Number of network locations', fontsize=14)
    ax.set_ylabel('Solution Quality (%)', fontsize=14)
    ax.set_title(title)
    ax.set_xticks(x + width, locs)
    ax.set_ylim(0, 140)

    # --- Now add Greedy Classical line (centered over grouped bars) ---
    if greedy_classical:
        ax.plot(
            # x + center_offset - 0.2,
            x + center_offset,
            greedy_classical,
            color='black',
            marker='D',
            linestyle='--',
            linewidth=1,
            markersize=5,
            label='Greedy Classical',
            zorder=3,
        )

    if AWS_results:
        ax.plot(
            x + center_offset + 0.3,
            AWS_results,
            # color='#9467bd',
            color=AWS_color,
            marker='X',
            linestyle='None',
            markersize=10,
            label='VQA: Rigetti Cepheus',
            zorder=2,
        )

        for xi, yi in zip(x, AWS_results):
            ax.annotate(
                f'{yi:.1f}',
                (xi, yi),
                xytext=AWS_xytext,
                textcoords='offset points',
                # color=AWS_color,
                color='black',
                fontsize=9,
                rotation=90,
            )

    if monte_carlo:
        ax.plot(
            x + center_offset,
            monte_carlo,
            color='#D81B60',  # magenta accent',
            marker='x',
            linestyle='--',
            linewidth=1,
            markersize=5,
            label='Monte Carlo comparison',
        )

    # --- Reorder legend so "Greedy Classical" appears last ---
    handles, labels = ax.get_legend_handles_labels()

    # Move 'Results from AWS', 'Greedy Classical' to the end

    if 'Greedy Classical' in labels:
        idx = labels.index('Greedy Classical')
        # Pop and append to end
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))

    if 'Monte Carlo comparison' in labels:
        idx = labels.index('Monte Carlo comparison')
        # Pop and append to end
        handles.append(handles.pop(idx))
        labels.append(labels.pop(idx))

    ax.legend(
        handles,
        labels,
        loc='upper right',
        ncols=n_cols,
        fontsize=legend_fontsize,
        framealpha=1,
        shadow='true',
    )
    filename = Path(GRAPH_DIR).joinpath('solution_quality_by_method.pdf')
    plt.savefig(filename, bbox_inches='tight')

    plt.show()


def plot_circuit_comparison(
    cma_stats: pd.DataFrame,
    vqa_mc_mean: pd.DataFrame,  # Monte Carlo mean
    vqa_mc_error: pd.DataFrame,  # Monte Carlo error
    locs: list,
):
    """plots comparisons between different circuits"""
    plt.figure()

    for mode, group in cma_stats.groupby('mode'):
        group_sorted = group.sort_values('locations')

        plt.errorbar(
            group_sorted['locations'],
            group_sorted['quality_mean'],
            yerr=group_sorted['quality_sem'],
            fmt='.',
            capsize=3,
            label=mode,
        )

    plt.errorbar(
        locs,
        vqa_mc_mean,
        yerr=vqa_mc_error,
        fmt='.',
        color='black',
        capsize=3,
        label='monte carlo',
    )

    # Labels and legend
    plt.xlabel('Locations')
    plt.ylabel('Quality')
    plt.title('Quality vs Locations by Mode')
    plt.legend(shadow='true')
    plt.show()


def plot_individual_run(df_detailed: pd.DataFrame):
    """plots results from individual runs"""
    plt.plot(
        df_detailed.index,
        df_detailed['average_list'],
        linewidth=1.0,
        color='blue',
        label='Simple average distance',
    )
    plt.plot(
        df_detailed.index,
        df_detailed['sliced_list'],
        linewidth=1.0,
        color='orange',
        label='Sliced average distance',
    )
    plt.plot(
        df_detailed.index,
        df_detailed['lowest_list'],
        linewidth=1.0,
        color='red',
        label='Lowest distance found so far',
    )
    plt.plot(
        df_detailed.index,
        df_detailed['best_dist'],
        linewidth=1.0,
        color='black',
        label='Lowest known distance',
    )
    title = 'Average_and_lowest_distance_found by_iteration'
    plt.title('Average_and_lowest_distance_found by_iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.legend()
    plt.tight_layout()

    filename = Path(GRAPH_DIR).joinpath(f'{title}.pdf')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_network(df: pd.DataFrame):
    """points out a network"""
    plt.scatter(df['x'], df['y'])

    # Add labels from 'location' column
    for i, row in df.iterrows():
        plt.text(row['x'], row['y'], str(row['Location']), fontsize=9, ha='right')

    # Axis labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot of x vs y with location labels')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_summary(setting: list, simulation_means: list):
    """Plot summary values in pretty colours"""
    # Colors
    colors = sns.color_palette('Set2', n_colors=len(simulation_means))

    # Create bar plot
    plt.figure(figsize=(8, 5))
    bars = plt.bar(setting, simulation_means, color=colors, edgecolor='black')

    # Labels and styling
    plt.xlabel('Setting', fontsize=12)
    plt.ylabel('Solution Quality (%)', fontsize=12)
    plt.title(
        'Solution Quality for Default, Warm Start, Factorial Formulation, and Gray Encoding',
        fontsize=13,
    )
    plt.ylim(0, 130)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f'{height:.1f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_cache_calls(
    locations: list, total_cache_calls: list, permutations: list, limit: float
):
    """plot total cache calls by location"""

    # Choose Set2 colors
    colors = sns.color_palette('Set2', n_colors=4)

    plt.bar(
        locations,
        total_cache_calls,
        color=colors[2],
        alpha=0.8,
        edgecolor='black',
        label='Average bit strings sampled',
    )
    plt.yscale('log')  # Log scale for the bar axis
    plt.ylabel('Average values (log scale)', fontsize=14)
    plt.tick_params(
        axis='y',
    )

    plt.plot(
        locations,
        permutations,
        marker='D',
        markersize=5,
        color='black',
        label='Distinct cycles',
    )

    plt.axhline(
        y=limit,
        color=colors[1],
        linestyle='--',
        linewidth=3,
        label='Maximum bit strings that could be sampled',
    )

    # --- Formatting ---
    title = 'Permutations_and_Total_Cache_Calls_by_Location'
    if PLOT_TITLE:
        plt.title(title, fontsize=18)
    plt.xlabel('Number of Locations', fontsize=14)
    plt.xticks(locations)
    plt.legend(
        loc='upper left',
    )

    # --- Layout adjustments ---
    plt.tight_layout()

    filename = Path(GRAPH_DIR).joinpath(f'{title}.pdf')
    plt.savefig(filename, bbox_inches='tight')

    plt.show()
