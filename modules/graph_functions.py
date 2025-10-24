import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from classes.MyModel import MySine
from pathlib import Path
import torch
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns

from modules.config import GRAPH_DIR, PLOT_TITLE

SPACE = ' '
CMAP = 'Set3'
CMAP_HEATMAP = 'cividis'

def parameter_graph(filename: str, 
                    title: str,
                    index_list: list, 
                    gradient_list: list, 
                    legend:list,
                    ):
    """Plots a graph of the parameter evolution by iteration."""
    p = plt.plot(index_list, gradient_list)
    plt.grid(axis='x')
    plt.legend(p, legend)
    plt.title(title)
    plt.tight_layout()
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value in radians')
    plt.savefig(filename)
    plt.show()

def find_size(cost_list: list)-> tuple:
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
        rows = int(length/2)
        columns = 2
    return(length, rows, columns)

def find_graph_statistics(av_list:np.ndarray, 
                          best:float)-> tuple:
    """Helper function for graph plotting to find good values for ymin and ymax"""
    maximum = float(np.max(av_list))
    ymin, ymax = int(best*.9), int(maximum*1.1) 
    return(ymin, ymax)

def find_best_coords(x_list:np.ndarray, 
                     best:float)->tuple:
    """Helper function for graph plotting to find coordinates for best known value line"""
    x1 =  [0, x_list[-1]]
    y1 =  [best, best]
    return(x1, y1)

def find_i_j(count:int, rows:int)->tuple:
    """Find indices for subplots"""        
    if count < rows:
        i, j = count, 0
    else:
        i, j = count - rows, 1
    return(i,j)

def cost_graph_multi(filename: str, 
                     parameter_list: list=None, 
                     x_list : list=None,
                     av_list :list=None, 
                     lowest_list: list=None,
                     sliced_list: list=None, 
                     best: float=None,
                     main_title: str='',
                     sub_title: str='',
                     x_label: str='Epoch',
                     figsize: tuple=(8,8),
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
        axs[i,j].plot(x_list, av_list[count], linewidth=1.0, color = 'blue', label='Average')
        if sliced_list:
            axs[i,j].plot(x_list, sliced_list[count], linewidth=1.0, color = 'orange', label='Sliced Average')
        axs[i,j].plot(x_list, lowest_list[count], linewidth=1.0, color = 'red', label='Lowest found')
        axs[i,j].plot(x1, y1, linewidth=1.0, color = 'black', label='Lowest known')
        axs[i,j].grid(axis='x', color='0.95')
        axs[i,j].axis(ymin=ymin, ymax=ymax)
        axs[i,j].set_xlabel(x_label, fontsize=6)
        axs[i,j].set_ylabel('Energy (Distance)', fontsize=6)
        axs[i,j].xaxis.set_tick_params(labelsize=7)
        axs[i,j].yaxis.set_tick_params(labelsize=7)
        if sub_title != '':
            sub_title_full = sub_title + f'{parameter_list[count]}'
            axs[i,j].set_title(sub_title_full, fontsize=6)
        axs[i,j].legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(filename)

def plot_shortest_routes(points: list, 
                         route1:list, 
                         route2:list = None
                         ):
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
            end_index = route2[i+1]
            plt.plot([x[start_index], x[end_index]], 
                     [y[start_index], y[end_index]], 
                     color='red')
        # Connect the last point back to the first to complete the cycle
        start_index = route2[-1]
        end_index = route2[0]
        plt.plot([x[start_index], x[end_index]], [y[start_index], y[end_index]], color='red')

    for i in range(len(route1) - 1):
        start_index = route1[i]
        end_index = route1[i+1]
        plt.plot([x[start_index], x[end_index]], 
                 [y[start_index], y[end_index]], 
                 color='blue')
        # Connect the last point back to the first to complete the cycle
        start_index = route1[-1]
        end_index = route1[0]
        plt.plot([x[start_index], x[end_index]], [y[start_index], y[end_index]], color='blue')

    red_patch = mpatches.Patch(color='red', label='The hot start route')
    blue_patch = mpatches.Patch(color='blue', label='The shortest route')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

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

def plot_3d_graph_models(grouped_means: pd.DataFrame, 
                         input: str,
                         input2: str = 'layers'
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
        x = loc_map[row['locations']] - dx/2    # Center the bar on the x-axis
        y = input2_map[row[input2]] - dy/2  # Center the bar on the y-axis
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

    legend_handles = [mpatches.Patch(color=input2_colors[layer], label=layer) for layer in input2_vals]
    plt.legend(handles=legend_handles, title=input2, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    formatted_input = input.replace('_', SPACE).lower()
    title = f'3D bar graph of {formatted_input} by {input2} and locations'
    if PLOT_TITLE:
        plt.title(title)
    filepath = Path(GRAPH_DIR).joinpath(f'{title}.pdf')

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight')
    plt.show()

def plot_3d_graph_slice(grouped_means: pd.DataFrame, 
                        input: str, 
                        show_sem:bool=False
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
    locations = grouped_means['locations'].unique()
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

    # Plot bars with different colors
    for i, row in grouped_means.iterrows():
        x = slice_map[row['slice']]   
        y = loc_map[row['locations']]
        x_bar = slice_map[row['slice']] - dx/2    # Center the bar on the x-axis
        y_bar = loc_map[row['locations']] - dy/2  # Center the bar on the y-axis
        z_bar = 0
        dz = row[input]
        if show_sem:
            error = row['sem']

        color = location_colors[row['locations']]
        ax.bar3d(x_bar , y_bar, z_bar, dx, dy, dz, color=color, shade=True)
        
        x_center = x + dx / 4
        y_center = y + dy / 4
        y_center = y

        if show_sem:
            if error > 0:
                #error bars
                ax.plot(
                    [x_center , x_center],
                    [y_center, y_center],
                    [0, dz + error],
                    color='black',
                    linewidth=2
                    )     
                # Horizontal cap at the top
                ax.plot(
                    [x_center - cap_width, x_center + cap_width],
                    [y_center, y_center],
                    [dz + error, dz + error],
                    color='black',
                    linewidth=2
                    )
    # Label axes
    ax.set_xlabel('Slice')
    ax.set_ylabel('Locations')
    ax.set_zlabel(input)

    # Set tick labels
    ax.set_xticks(list(slice_map.values()))
    ax.set_xticklabels(list(slice_map.keys()))
    ax.set_yticks(list(loc_map.values()))
    ax.set_yticklabels(list(loc_map.keys()))

    legend_handles = [mpatches.Patch(color=location_colors[loc], label=loc) for loc in locations]
    plt.legend(handles=legend_handles, title='Locations', loc='upper left', bbox_to_anchor=(1, 1))

    formatted_input = input.replace('_', SPACE).lower()
    title = f'3D bar graph of {formatted_input} by location and slice'
    if PLOT_TITLE:
        plt.title(title)
    filepath = Path(GRAPH_DIR).joinpath(f'{title}.pdf')

    plt.savefig(filepath, bbox_inches='tight')

    plt.show()

def plot_heatmap(input: pd.DataFrame,
                 title: str,
                 x_label:str,
                 y_label:str,
                 )-> None:
    """Plot a heat map of the given input data."""
    # Prepare the data
    heatmap_data = input.values
    x_labels = input.columns.format()
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
    for i in range(heatmap_data.shape[0]):       # rows
        for j in range(heatmap_data.shape[1]):   # columns
            value = heatmap_data[i, j]
            if not np.isnan(value):  # skip missing values
                if value > 80:
                    color = 'black'
                else:
                    color = 'white'
                plt.text(j, i, f"{value:.1f}", ha='center', va='center', color=color)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.tight_layout()
    plt.show()