import matplotlib.pyplot as plt
import numpy as np

def get_depths_order_and_labels(root, counts):
    """This function assigns a depth to each node based on how many unary or
    binary operations it takes to get to it from variables or constants.  This
    also assigns an order to the node, for the time that it appears at a certain
    depth in postorder.  It also computes appropriate labels for the nodes.
    This is used to compute render coordinates for visualization.

    arguments:
    root -- the root node to start from
    count -- a list of integers for the count of nodes at each depth
    """
    if root:
        get_depths_order_and_labels(root.left, counts)
        get_depths_order_and_labels(root.right, counts)
        if not root.left and not root.right:
            root.depth = 0
        elif not root.right:
            root.depth = root.left.depth + 1
        else:
            root.depth = max(root.left.depth, root.right.depth) + 1

        if len(counts) <= root.depth:
            counts.append(1)
        else:
            counts[root.depth]+=1

        root.order = counts[root.depth] - 1
        if root.depth > 0:
            root.label = f'$V_{{{root.depth},{root.order+1}}}$'
        else:
            if root.type == 'const':
                root.label = root.value
            else:
                root.label = root.var_name

def get_node_positions(root, counts, max_count):
    """This function computes node positions given the calculated
    depths and order information.

    arguments:
    root -- the root node to start from
    count -- a list of integers for the count of nodes at each depth
    """

    if root:
        get_node_positions(root.left, counts, max_count)
        get_node_positions(root.right, counts, max_count)

        max_order = counts[root.depth]
        y_min = (max_count - max_order)/2.0

        root.plot_x = root.depth
        root.plot_y = root.order + y_min

def render_edges(root, fontsize = 16):
    """This function renders the edges of the nodes in ourgraph.

      arguments:
      root -- the root node to start from
    """
    if root:
        render_edges(root.left, fontsize=fontsize)
        render_edges(root.right, fontsize=fontsize)

        if root.left:
            plt.plot( (root.plot_x, root.left.plot_x),
                      (root.plot_y, root.left.plot_y), zorder=-1, c='w', lw=1)
            if root.function:
                # add a label
                label_x = .6 * root.plot_x + .4 * root.left.plot_x
                label_y = .6 * root.plot_y + .4 * root.left.plot_y
                plt.text(label_x, label_y, root.function_name, fontdict={'size': fontsize, 'color': 'green'},
                         bbox={'facecolor': 'black', 'alpha': 0.9, 'edgecolor': 'gray', 'pad': 1},
                         ha='center', va='center')


        if root.right:
            plt.plot( (root.plot_x, root.right.plot_x),
                      (root.plot_y, root.right.plot_y), zorder=-1, c='w', lw=1)
            if root.function:
                # add a label
                label_x = .6 * root.plot_x + .4 * root.right.plot_x
                label_y = .6 * root.plot_y + .4 * root.right.plot_y
                plt.text(label_x, label_y, root.function_name, fontdict={'size': fontsize, 'color': 'green'},
                         bbox={'facecolor': 'black', 'alpha': 0.9, 'edgecolor': 'gray', 'pad': 1},
                         ha='center', va='center')

def render_values(root, fontsize=16):
    """This function renders the nodes at their calculated positions based on their
    type.

    arguments:
    root -- the root node to start from
    """
    if root:
        render_values(root.left, fontsize=fontsize)
        render_values(root.right, fontsize=fontsize)

        if root.value:
            if isinstance(root.value, int):
                text = f'={root.value}'
            else:
                text = f'={root.value:.3f}'

            plt.text(root.plot_x + .2, root.plot_y - .2, text, fontdict={'size':fontsize, 'color':'yellow'},
                     bbox={'facecolor':'black','alpha':0.7,'edgecolor':'white','pad':1},
                     ha='center', va='center')

def render_points(root, fontsize=16):
    """This function renders the nodes at their calculated positions based on their
    type.

    arguments:
    root -- the root node to start from
    """
    if root:
        render_points(root.left, fontsize=fontsize)
        render_points(root.right, fontsize=fontsize)

        if root.type == 'inter':
            shape = plt.Circle( (root.plot_x, root.plot_y), radius=0.2,
                                 fc='blue', ec='lightblue', lw=1)
        elif root.type == 'var':
            shape = plt.Rectangle((root.plot_x-.2, root.plot_y-.2),
                                  .4, .4, fc='darkred', ec='pink', lw=1)
        else:
            shape = plt.Rectangle((root.plot_x-.2, root.plot_y-.2),
                                  .4, .4, fc='purple', ec='pink', lw=1)
        plt.gca().add_patch(shape)
        plt.text(root.plot_x, root.plot_y, root.label, fontdict={'size':fontsize, 'color':'white'},
                 bbox={'facecolor':'black','alpha':0.7,'edgecolor':'gray','pad':1},
                 ha='center', va='center')

def prepare_plot(depth_counts):
    fig = plt.figure( figsize = (10, 10))
    fig.patch.set_facecolor('black')
    plt.rc('axes', edgecolor='darkgray')
    plt.axis('off')

    maxx = len(depth_counts)
    maxy = np.max(depth_counts)
    plt.gca().set_xlim(-0.5, maxx - 0.5)
    plt.gca().set_ylim(-0.5, maxy - 0.5)
    plt.gca().set_facecolor('black')

    max_order = np.max(depth_counts)
    max_depth = len(depth_counts)

    # let's draw a background grid just for style
    for i in np.arange(0, max_depth - 0.75, 0.5):
        plt.plot((i,i), (0, max_order-1), c=(0.1, 0.1, 0.1),
                 zorder=-2)

    for i in np.arange(0, max_order - 0.75, 0.5):
        plt.plot((0, max_depth-1), (i, i), c=(0.1, 0.1, 0.1),
                 zorder=-2)

    fontsize = 18 - len(depth_counts)
    print (fontsize)
    return fontsize

def conclude_plot():
    plt.show()



