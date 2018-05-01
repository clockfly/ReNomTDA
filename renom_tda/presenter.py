# -*- coding: utf-8 -*.t-
from __future__ import print_function
from abc import ABCMeta, abstractmethod
from future.utils import with_metaclass
from copy import copy
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Presenter(with_metaclass(ABCMeta, object)):
    """Abstract class of present topology."""

    def is_type(self, mode):
        """check mode."""
        if mode == self.mode:
            return True
        return False

    @abstractmethod
    def show(self):
        """show function is required."""
        pass

    @abstractmethod
    def save(self):
        """save function is required."""
        pass


class NormalPresenter(Presenter):
    """Class of present topology with node coordinate.

    Params:
        fig_size: The size of figure.

        node_size: The size of node.

        edge_width: The width of edge.
    """

    def __init__(self, fig_size, node_size, edge_width):
        self.mode = "normal"
        self.fig_size = fig_size
        self.node_size = node_size
        self.edge_width = edge_width

    def _plot(self, nodes, edges, node_sizes, colors):
        """Function of create figure."""
        fig = plt.figure(figsize=self.fig_size)
        ax = fig.add_subplot(111)
        for e in edges:
            ax.plot([nodes[e[0], 0], nodes[e[1], 0]],
                    [nodes[e[0], 1], nodes[e[1], 1]],
                    c=colors[e[0]], lw=self.edge_width, zorder=1)
        ax.scatter(nodes[:, 0], nodes[:, 1], c=colors, s=node_sizes * self.node_size, zorder=2)
        plt.axis("off")

    def show(self, nodes, edges, node_sizes, colors):
        """Function of show figure.

        Params:
            nodes: array of node coordinates.

            edges :array of edges.

            node_sizes: array of node sizes.

            colors: array of node colors.
        """
        self._plot(nodes, edges, node_sizes, colors)
        plt.show()

    def save(self, file_name, nodes, edges, node_sizes, colors):
        """Function of show figure.

        Params:
            file_name: output file name.

            nodes: array of node coordinates.

            edges :array of edges.

            node_sizes: array of node sizes.

            colors: array of node colors.
        """
        self._plot(nodes, edges, node_sizes, colors)
        plt.savefig(file_name)


class SpringPresenter(Presenter):
    """Class of present topology with spring force layout.

    Params:
        fig_size: The size of figure.

        node_size: The size of node.

        edge_width: The width of edge.

        strength: strength of repulsive between nodes.
    """

    def __init__(self, fig_size, node_size, edge_width, strength):
        self.mode = "spring"
        self.fig_size = fig_size
        self.node_size = node_size
        self.edge_width = edge_width
        self.strength = strength

    def _calc_init_position(self, nodes):
        """calculate node initial position."""
        init_position = {}

        if nodes.shape[1] == 1:
            for i, n in enumerate(nodes):
                init_position.update({i: [n[i], n[i]]})
        elif nodes.shape[1] == 2 or nodes.shape[1] == 3:
            for i, n in enumerate(nodes):
                print(n)
                init_position.update({i: n[:2]})
        return init_position

    def _plot(self, nodes, edges, node_sizes, colors):
        """Function of create figure."""
        init_position = self._calc_init_position(nodes)

        # with networkx spring layout.
        fig = plt.figure(figsize=self.fig_size)
        fig.add_subplot(111)
        graph = nx.Graph(pos=init_position)
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from(edges)
        position = nx.spring_layout(graph, pos=init_position, k=self.strength)

        nx.draw_networkx(graph, pos=position,
                         node_size=node_sizes * self.node_size,
                         node_color=colors,
                         width=self.edge_width,
                         edge_color=[colors[e[0]] for e in edges],
                         with_labels=False)
        plt.axis("off")

    def show(self, nodes, edges, node_sizes, colors):
        """Function of show figure.

        Params:
            nodes: array of node coordinates.

            edges :array of edges.

            node_sizes: array of node sizes.

            colors: array of node colors.
        """
        self._plot(nodes, edges, node_sizes, colors)
        plt.show()

    def save(self, file_name, nodes, edges, node_sizes, colors):
        """Function of show figure.

        Params:
            file_name: output file name.

            nodes: array of node coordinates.

            edges :array of edges.

            node_sizes: array of node sizes.

            colors: array of node colors.
        """
        self._plot(nodes, edges, node_sizes, colors)
        plt.savefig(file_name)


class SpectralPresenter(Presenter):
    """Class of spectral graph clustering plot.

    Params:
        fig_size: The size of figure.

        node_size: The size of node.

        edge_width: The width of edge.

        strength: strength of repulsive between nodes.
    """

    def __init__(self, fig_size, node_size, edge_width, strength):
        self.mode = "spectral"
        self.fig_size = fig_size
        self.node_size = node_size
        self.edge_width = edge_width
        self.strength = strength

    def _plot(self, nodes, edges, node_sizes, colors):
        # init adjacency matrix
        adjacency_matrix = np.zeros((len(nodes), len(nodes)))

        # create adjacency_matrix edges
        for e in edges:
            adjacency_matrix[e[0], e[1]] = 1
            adjacency_matrix[e[1], e[0]] = 1

        # copy adjacency matrix
        # 1 cluster nodes regard cluster
        dummy_matrix = copy(adjacency_matrix)

        # independent nodes
        independent_index = np.where(np.sum(adjacency_matrix, axis=0) == 0)[0]

        # make dummy edge
        for i in itertools.permutations(independent_index, 2):
            dummy_matrix[i[0], i[1]] = 1
            dummy_matrix[i[1], i[0]] = 1

        # make graph from dummy matrix
        dummy_graph = nx.from_numpy_matrix(dummy_matrix)
        # graph connections
        connections = [set(x) for x in nx.connected_components(dummy_graph)]

        # cluster size dict
        cluster = {}
        for i, c in enumerate(nx.connected_components(dummy_graph)):
            cluster.update({i: len(c)})
        cluster = sorted(cluster.items(), key=lambda x: x[1], reverse=True)

        # cluster max size
        c_max = cluster[0][1]
        # init row clusters list
        arange_list = []

        # tmp data
        tmp_size_sum = 0
        tmp_index_list = []
        # append cluster index list to arange_list per row
        for i, c in cluster:
            tmp_index_list.append(i)
            tmp_size_sum += c
            if tmp_size_sum >= c_max:
                arange_list.append(tmp_index_list)
                tmp_size_sum = 0
                tmp_index_list = []
        arange_list.append(tmp_index_list)

        # calc coordinate from dummy graph
        pos = {}
        for ind, l in enumerate(arange_list):
            accum_cluster_size = 0
            total_cluster_size = c_max

            # calc coordinate per clusters
            for i in l:
                data_index = np.array(list(connections[i]))
                cluster_size = len(connections[i])
                subcluster = dummy_matrix[data_index][:, data_index]

                init_pos = {}
                for j, n in enumerate(nodes[data_index]):
                    init_pos.update({j: [n[0], n[1]]})

                g = nx.from_numpy_matrix(subcluster)
                p = nx.spring_layout(g, pos=init_pos, k=self.strength)

                tmp = cluster_size / total_cluster_size
                for k in p.keys():
                    pos[data_index[k]] = p[k] * [tmp, 1 / len(arange_list)] + [accum_cluster_size, 1 / len(arange_list) * ind]
                accum_cluster_size += tmp

        # calc spring layout
        g = nx.from_numpy_matrix(dummy_matrix)
        pos = nx.spring_layout(g, pos=pos, k=self.strength)

        plt.figure(figsize=(15, 10))
        g = nx.from_numpy_matrix(adjacency_matrix)
        nx.draw_networkx(g, pos=pos, node_size=node_sizes * 5, node_color=colors,
                         edge_color=[colors[e[0]] for e in edges], with_labels=False)
        plt.axis("off")

    def show(self, nodes, edges, node_sizes, colors):
        self._plot(nodes, edges, node_sizes, colors)
        plt.show()

    def save(self, file_name, nodes, edges, node_sizes, colors):
        self._plot(nodes, edges, node_sizes, colors)
        plt.savefig(file_name)


class PresenterResolver(object):
    """Resolve presenter from present mode."""

    def __init__(self, fig_size, node_size, edge_width, strength):
        self.presenter_list = [
            NormalPresenter(fig_size, node_size, edge_width),
            SpringPresenter(fig_size, node_size, edge_width, strength),
            SpectralPresenter(fig_size, node_size, edge_width, strength)]

    def resolve(self, mode):
        """Function of getting presenter.

        Params:
            mode: present mode.

        Return:
            presenter: resolved presenter instance.
        """
        for p in self.presenter_list:
            if p.is_type(mode):
                return p
