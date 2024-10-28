import matplotlib.pyplot as plt

def parameter_graph(filename, index_list, gradient_list, legend):
    p = plt.plot(index_list, gradient_list)
    plt.grid(axis='x')
    plt.legend(p, legend)
    plt.title('Evolution of parameters with iterations')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def cost_graph(filename, index_list, cost_list, lowest_list):
    plt.style.use('seaborn-v0_8-colorblind')
    plt.plot(index_list, cost_list, linewidth=1.0, label='Average')
    plt.step(index_list, lowest_list, linewidth=1.0, color = 'red', label='Lowest')
    plt.grid(axis='x', color='0.95')
    plt.title('Average distance and lowest distance found')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()