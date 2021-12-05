import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import numpy as np

def render_first_frame(root):
    """This function renders the first frame of our animation.  For the first frame, the node depths and
        positions for plotting in the binary tree are calculated, we create a matplotlib figure for animation,
        and prepare for future frames.  The node depths and positions are stored in each node object.

       arguments:
       root -- the root of our binary tree
       """


    print ('rendering forward computation graph')

    depth_counts = []
    get_depths_order_and_labels(root, depth_counts)
    get_node_positions(root, depth_counts, np.max(depth_counts))

    num_nodes = np.sum(depth_counts)
    print(f'with {num_nodes} nodes')
    print ('-'*(num_nodes))

    fig, font_size = prepare_plot(depth_counts)
    plt.cla()
    render_grid(depth_counts)
    render_edges(root, font_size=font_size)
    render_points(root, font_size=font_size)
    render_values(root, font_size=font_size)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(1000, 1000, 3)
    return image, fig, font_size, depth_counts

def frame(root, fig, font_size, depth_counts, visit):
    """This function renders a frame of our animation based on the given binary tree, the current
       visit node, the figure generated in prepare_plot, the depth_counts, the current visit node
       and the node positions that were cxalculated.

       arguments:
       root -- the root of our binary tree
       fig -- a matplotlib figure object
       font_size -- the font size to use for this frame.  This is used to render the edges, values, and nodes
       depth_counts -- a list containing the number of nodes at each depth
       visit -- The current node to render the legend values for.  The value and all
       relevant derivatives are taken from the node attributes.
       """


    images = []
    print('.', end='')
    sys.stdout.flush()
    for inner_frame in np.linspace(0, 1, 4):
        plt.cla()
        render_grid(depth_counts, visit)
        render_edges(root, font_size=font_size, visit = visit, inner_frame = inner_frame)
        render_points(root, font_size=font_size, visit = visit)
        render_values(root, font_size=font_size, visit = visit, inner_frame = inner_frame)
        render_legend(visit)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(1000, 1000, 3)
        images.append(image)
    return images

def render_legend(visit):
    """This function renders a legend showing the current value and derivatives at the given
      'visit' node.  As the binary tree traversal occurs, the value and the derivatives are updated
      to show the current values.

       arguments:
       visit -- The current node to render the legend values for.  The value and all
       relevant derivatives are taken from the node attributes.
       """
    if visit.deriv:
        legend_entries = [f'df/d{key} = {value:.3f}' for key, value in visit.deriv.items()]
        legend_entries.append(f'value: {visit.value:0.3f}')
        legend = plt.legend(legend_entries, loc = 0, frameon = True)
        frame = legend.get_frame()
        frame.set_facecolor('black')
        frame.set_edgecolor('lightblue')
        for e, text in enumerate(legend.get_texts()):
            if e == len(legend_entries) - 1:
                text.set_color("yellow")
            else:
                text.set_color("pink")

        values_to_display = list(visit.deriv.values())
        values_to_display.append(visit.value)
        for handle, value in zip(legend.legendHandles, values_to_display):

            if value == 0:
                handle.set_color('gray')
            else:
                handle.set_color('lightblue')



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
    depths and order information.  We create a custom node position
    based on a partial ordering that is computed from the maximum
    depth of a node to a constant or vector.

    arguments:
    root -- the root node to start from
    count -- a list of integers for the count of nodes at each depth
    max_count -- the maximum count of nodes in any depth.  This is used to determine
    the ideal positions of the nodes within each depth.  We attempt to scatter the nodes
    so that they don't get in the way of edges.
    """

    if root:
        get_node_positions(root.left, counts, max_count)
        get_node_positions(root.right, counts, max_count)

        max_order = counts[root.depth]
        y_min = (max_count - max_order)/2.0

        root.plot_x = root.depth
        root.plot_y = root.order + y_min


def render_edges(root, font_size = 16, visit=None, inner_frame = 1.0):
    """This function renders the edges of the nodes in our graph.  These are animated
    based on the current node 'visit' and the inner frame from 0 to 1 to show movement
    along the graph.

      arguments:
      root -- the root node to start rendering the binary tree from
      font_size -- the font size to use for the render.  More complex functions require smaller
                   font sizes for the labeled functions.
      visit -- the currently visited node.  This the child edges attached to this node are
               highlighted and animated based on the inner frame value
      inner_frame -- a value from 0 to 1 used for animating the edges
    """
    if root:
        render_edges(root.left, font_size=font_size, visit = visit, inner_frame = inner_frame)
        render_edges(root.right, font_size=font_size, visit = visit, inner_frame = inner_frame)

        if root == visit:
            lfont_size = font_size + 2
        else:
            lfont_size = font_size

        if root.value:
            edge_color = 'pink'
            edge_weight = 3
        else:
            edge_color = 'w'
            edge_weight = 1

        if root.left:
            plt.plot( (root.plot_x, root.left.plot_x),
                      (root.plot_y, root.left.plot_y), zorder=-1, c=edge_color, lw=edge_weight)

            if root == visit:
                plt.plot( (root.left.plot_x * (1-inner_frame) + root.plot_x * (inner_frame), root.left.plot_x),
                      (root.left.plot_y * (1-inner_frame) + root.plot_y * (inner_frame), root.left.plot_y),
                      zorder = -0.5, c = 'yellow', lw = 6)

            if root.function:
                # add a label
                label_x = .6 * root.plot_x + .4 * root.left.plot_x
                label_y = .6 * root.plot_y + .4 * root.left.plot_y
                plt.text(label_x, label_y, root.function_name, fontdict={'size': lfont_size, 'color': 'green'},
                         bbox={'facecolor': 'black', 'alpha': 0.9, 'edgecolor': 'gray', 'pad': 1},
                         ha='center', va='center')


        if root.right:
            plt.plot( (root.plot_x, root.right.plot_x),
                      (root.plot_y, root.right.plot_y), zorder=-1, c=edge_color, lw=edge_weight)

            if root == visit:
                plt.plot( (root.right.plot_x * (1-inner_frame) + root.plot_x * (inner_frame), root.right.plot_x),
                      (root.right.plot_y * (1-inner_frame) + root.plot_y * (inner_frame), root.right.plot_y),
                      zorder = -0.5, c = 'yellow', lw = 6)

            if root.function:
                # add a label
                label_x = .6 * root.plot_x + .4 * root.right.plot_x
                label_y = .6 * root.plot_y + .4 * root.right.plot_y
                plt.text(label_x, label_y, root.function_name, fontdict={'size': lfont_size, 'color': 'green'},
                         bbox={'facecolor': 'black', 'alpha': 0.9, 'edgecolor': 'gray', 'pad': 1},
                         ha='center', va='center')

def render_values(root, font_size=16, visit = None, inner_frame = 1.0):
    """This function renders the nodes at their calculated positions based on their
    type.

    arguments:
    root -- the root node to start from
    font_size -- the font size to render the values with.  Larger functions need to use a smaller font.
    visit -- the current node to highlight
    inner_frame -- the sub frame of the animation used to render highlighted lines.  The value will only
                    displayed when the inner frame is close to 1.
    """
    if root:
        render_values(root.left, font_size=font_size, visit = visit, inner_frame = inner_frame)
        render_values(root.right, font_size=font_size, visit = visit, inner_frame = inner_frame)

        if root.value:
            if not (visit == root and inner_frame < 0.95):
                if isinstance(root.value, int):
                    text = f'={root.value}'
                else:
                    text = f'={root.value:.3f}'

                plt.text(root.plot_x + .2, root.plot_y - .2, text, fontdict={'size':font_size, 'color': 'yellow'},
                         bbox={'facecolor':'black','alpha':0.7,'edgecolor':'white','pad':1},
                         ha='center', va='center')

def render_points(root, font_size=16, visit = None):
    """This function renders the nodes at their calculated positions based on their
    type.

    arguments:
    root -- the root node to start from
    font_size -- optionally specify the font size to use. For larger functions, we want
                 to use a smaller font size
    visit -- the current node that we are visiting.  This node will be highlighted.
    """
    if root:
        render_points(root.left, font_size=font_size, visit = visit)
        render_points(root.right, font_size=font_size, visit = visit)

        if root == visit:
            color = 'yellow'
        if root.type == 'inter':
            if root != visit:
                color = 'blue'
            shape = plt.Circle( (root.plot_x, root.plot_y), radius=0.2,
                                 fc=color, ec='lightblue', lw=1)
        elif root.type == 'var':
            if root != visit:
                color = 'darkred'
            shape = plt.Rectangle((root.plot_x-.2, root.plot_y-.2),
                                  .4, .4, fc=color, ec='pink', lw=1)
        else:
            if root != visit:
                color = 'purple'
            shape = plt.Rectangle((root.plot_x-.2, root.plot_y-.2),
                                  .4, .4, fc=color, ec='pink', lw=1)
        plt.gca().add_patch(shape)
        plt.text(root.plot_x, root.plot_y, root.label, fontdict={'size':font_size, 'color': 'white'},
                 bbox={'facecolor':'black','alpha':0.7,'edgecolor':'gray','pad':1},
                 ha='center', va='center')

def prepare_plot(depth_counts):
    """This function prepares our plot for rendering.  It creates a figure, turns off the axes,
    and sets the appropriate limits based on the depth counts.

    arguments:
    depth_counts -- a list of the number of nodes per each depth
    """
    fig = plt.figure( figsize = (10, 10), dpi=100)
    fig.patch.set_facecolor('black')
    plt.rc('axes', edgecolor='darkgray')
    plt.axis('off')

    maxx = len(depth_counts)
    maxy = np.max(depth_counts)
    plt.gca().set_xlim(-0.75, maxx - 0.75)
    plt.gca().set_ylim(-0.5, maxy - 0.5)
    plt.gca().set_facecolor('black')

    fontsize = 18 - len(depth_counts)
    return fig, fontsize

def render_grid(depth_counts, visit=None):
    """This function renders the background grid for our forward graph.  We
    use the depth counts, the number of nodes per each depth to determine the
    grid placement.  If visis is not None, we highlight that particular row
    and column.

    arguments:
    depth_counts -- a list of the number of nodes per each depth
    visit -- the current visited node used for grid highlighting
    """

    max_order = np.max(depth_counts)
    max_depth = len(depth_counts)

    if visit:
        visit_x = visit.plot_x
        visit_y = visit.plot_y

    # let's draw a background grid just for style
    for i in np.arange(0, max_depth - 0.75, 0.5):
        if visit and i == visit_x:
            color = (0.6, 0.2, 0.3  )
        else:
            color = (0.1, 0.1, 0.1)

        plt.plot((i, i), (0, max_order - 1), c=color,
                 zorder=-2)

    for i in np.arange(0, max_order - 0.75, 0.5):
        if visit and i == visit_y:
            color = (0.6, 0.2, 0.3)
        else:
            color = (0.1, 0.1, 0.1)

        plt.plot((0, max_depth - 1), (i, i), c=color,
                 zorder=-2)




