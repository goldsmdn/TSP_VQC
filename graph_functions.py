import matplotlib.pyplot as plt

def parameter_graph(filename: str, index_list: list, gradient_list:list, legend:list):
    """plots a graph of the parameter evolution"""
    p = plt.plot(index_list, gradient_list)
    plt.grid(axis='x')
    plt.legend(p, legend)
    plt.title('Evolution of parameters with iterations')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def cost_graph(filename: str, index_list: list, cost_list: list, lowest_list: list, title: str=''):
    """plots a graph of the cost function by interation"""
    plt.style.use('seaborn-v0_8-colorblind')
    plt.plot(index_list, cost_list, linewidth=1.0, label='Average')
    plt.step(index_list, lowest_list, linewidth=1.0, color = 'red', label='Lowest')
    plt.grid(axis='x', color='0.95')
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def param_cost_graph(filename: str, x_list :list, cost_list :list, 
                     lowest_list: list,sliced_list: list, best: float):
    """plots a graph of the cost functions against the parameters"""
    length = len(cost_list)
    if length % 2 != 0:
        raise Exception(f'For this plot the number of lists must be even, but the number of lists - {length} is odd')
    rows = int(length/2)
    columns = 2
    fig, axs = plt.subplots(rows, columns, figsize=(8,8))
    fig.suptitle('Average and lowest energy (distance) found by changing each parameter in turn')
    #calculated points for best possible line
    x1 =  [0, x_list[-1]]
    y1 =  [best, best]
    ymin, ymax = int(best-1), int(best*1.5) 

    for count in range(length):
        if count < rows:
            i, j = count, 0
        else:
            i, j = count - rows, 1
        axs[i,j].plot(x1, y1, linewidth=1.0, color = 'black', label='Lowest possible')
        axs[i,j].step(x_list, lowest_list[i], linewidth=1.0, color = 'red', label='Lowest found')
        axs[i,j].plot(x_list, cost_list[i], linewidth=1.0, color = 'blue', label='Average')
        axs[i,j].plot(x_list, sliced_list[i], linewidth=1.0, color = 'orange', label='Sliced Average')
        axs[i,j].grid(axis='x', color='0.95')
        axs[i,j].axis(ymin=ymin, ymax=ymax)
        axs[i,j].set_xlabel('Gate rotation in Radians', fontsize=6)
        axs[i,j].set_ylabel('Energy (Distance)', fontsize=6)
        axs[i,j].xaxis.set_tick_params(labelsize=7)
        axs[i,j].yaxis.set_tick_params(labelsize=7)
        title = f'Parameter {count}'
        axs[i,j].set_title(title, fontsize=6)
        axs[i,j].legend(fontsize=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(filename)
    fig.show()