import networkx as nx
import matplotlib.pyplot as plt


# Draw a graph from a matrix with colors for each player
# and a label for each territory and troops count on it
def draw_graph(map_graph, colors, labels):
    G = nx.from_numpy_array(map_graph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels)
    plt.show()
