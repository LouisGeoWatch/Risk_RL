import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os


# Draw a graph from a matrix with colors for each player
# and a label for each territory and troops count on it
def draw_graph(map_graph, colors, labels):
    G = nx.from_numpy_array(map_graph)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors, with_labels=True, labels=labels)
    plt.show()


def draw_map(map_graph, presence_map):

    G = nx.from_numpy_array(map_graph)

    owner = np.argmax(presence_map, axis=0)
    troops = np.max(presence_map, axis=0)

    colors = {0: 'red', 1: 'blue', 2: 'green'}
    colors = [colors[i] for i in owner]

    countries = {0: "N. America",
                 1: "S. America",
                 2: "Europe",
                 3: "Africa",
                 4: "Asia",
                 5: "Oceania"
                 }

    layout = {0: np.array([0.5, 0.5]),
              1: np.array([0.5, 0.3]),
              2: np.array([0.5, 0.7]),
              3: np.array([0.7, 0.5]),
              4: np.array([0.3, 0.5]),
              5: np.array([0.3, 0.3])
              }

    label_options = {"ec": "k", "fc": "white", "alpha": 0.3}
    labels = {i: f"{i}: {countries[i]} ({troops[i]})" for i in range(map_graph.shape[0])}

    plt.figure(figsize=(6, 3))
    plt.margins(x=0.2)
    nx.draw(G, layout, node_color=colors, labels=labels, bbox=label_options)
    plt.show()


def draw_map_and_save(map_graph, presence_map, title=None, filename=None):
    plt.figure()

    G = nx.from_numpy_array(map_graph)

    owner = np.argmax(presence_map, axis=0)
    troops = np.max(presence_map, axis=0)

    colors = {0: 'red', 1: 'blue', 2: 'green'}
    colors = [colors[i] for i in owner]

    countries = {0: "N. America",
                 1: "S. America",
                 2: "Europe",
                 3: "Africa",
                 4: "Asia",
                 5: "Oceania"
                 }

    layout = {0: np.array([0.5, 0.5]),
              1: np.array([0.5, 0.3]),
              2: np.array([0.5, 0.7]),
              3: np.array([0.7, 0.5]),
              4: np.array([0.3, 0.5]),
              5: np.array([0.3, 0.3])
              }

    label_options = {"ec": "k", "fc": "white", "alpha": 0.3}
    labels = {i: f"{i}: {countries[i]} ({troops[i]})" for i in range(map_graph.shape[0])}

    plt.figure(figsize=(6, 3))
    plt.margins(x=0.2)
    if title is not None:
        plt.title(title)
    nx.draw(G, layout, node_color=colors, labels=labels, bbox=label_options)

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

    plt.close()


def generate_gif(folder_path, output_path, fps=1):

    # Sort the images in the folder by name
    images = sorted(os.listdir(folder_path))
    # Create a list of image paths
    image_paths = [os.path.join(folder_path, img) for img in images]
    # Read the images into a list of arrays
    image_arrays = [imageio.imread(img_path) for img_path in image_paths]
    # Save the list of arrays as a GIF
    imageio.mimsave(os.path.join(output_path, 'output.gif'), image_arrays, fps=fps)
    # Delete the images
    for img_path in image_paths:
        os.remove(img_path)
