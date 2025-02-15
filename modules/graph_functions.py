import matplotlib.pyplot as plt

def parameter_graph(filename: str, index_list: list, gradient_list:list, legend:list):
    """plots a graph of the parameter evolution"""
    p = plt.plot(index_list, gradient_list)
    plt.grid(axis='x')
    plt.legend(p, legend)
    plt.title('Evolution of parameters with iterations')
    plt.tight_layout()
    plt.xlabel('Iteration')
    plt.ylabel('Parameter value in radians')
    plt.savefig(filename)
    plt.show()

def find_size(cost_list):
    """find the number of subplots"""
    length = len(cost_list)
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
    ymin, ymax = int(best-1), int(best*1.5) 
    return(x1, y1, ymin, ymax)

def find_i_j(count, rows):
    """find indices for subplots"""        
    if count < rows:
        i, j = count, 0
    else:
        i, j = count - rows, 1
    return(i,j)

def cost_graph_multi(filename: str, 
                     parameter_list: list, 
                     x_list : list,
                     av_list :list, 
                     lowest_list: list,
                     sliced_list: list, 
                     best: float,
                     main_title: str='',
                     sub_title: str='',
                     x_label: str='',
                     figsize: tuple=(8,8)
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