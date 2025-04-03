import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from classes.MyModel import MySine
from pathlib import Path
from modules.config import GRAPH_DIR
import torch

def parameter_graph(filename: str, 
                    title: str,
                    index_list: list, 
                    gradient_list:list, 
                    legend:list):
    """plots a graph of the parameter evolution"""
    p = plt.plot(index_list, gradient_list)
    plt.grid(axis='x')
    plt.legend(p, legend)
    plt.title(title)
    plt.tight_layout()
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value in radians')
    plt.savefig(filename)
    plt.show()

def find_size(cost_list):
    """find the number of subplots"""
    if cost_list:
        length = len(cost_list)
    else:
        length = 1
    if length == 1:
        rows = 1    
        columns = 1
    elif length % 2 != 0:
        raise Exception(f'For this plot the number of lists must be even, but the number of lists - {length} is odd')
    else:
        rows = int(length/2)
        columns = 2
    return(length, rows, columns)

def find_graph_statistics(x_list, best):
    """helper function for graph plotting"""
    x1 =  [0, x_list[-1]]
    y1 =  [best, best]
    ymin, ymax = int(best-1), int(best*2) 
    return(x1, y1, ymin, ymax)

def find_i_j(count, rows):
    """find indices for subplots"""        
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
                     x_label: str='',
                     figsize: tuple=(8,8),
                     ):
    """plots a graph of the cost function for multiple lists"""
    plt.style.use('seaborn-v0_8-colorblind')
    length, rows, columns = find_size(parameter_list)
    fig, axs = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
    fig.suptitle(main_title)
    x1, y1, ymin, ymax = find_graph_statistics(x_list, best)

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

def plot_shortest_routes(points, route1, route2=None):
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
    title = 'Sine_Activation_Function'
    filepath = Path(GRAPH_DIR).joinpath(title + '.png')

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
    plt.savefig(filepath)
    plt.show()

"""def plot_model_training(epoch_history, loss_history):
    title = 'Loss_by_epoch'
    filepath = Path(GRAPH_DIR).joinpath(title + '.png')
    plt.plot(epoch_history, loss_history)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(filepath)
    plt.show"""