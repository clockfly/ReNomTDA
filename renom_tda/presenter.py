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
    """

    def __init__(self, fig_size, node_size, edge_width):
        self.mode = "spectral"
        self.fig_size = fig_size
        self.node_size = node_size
        self.edge_width = edge_width

    def _init_adjacency_matrix(self, nodes, edges):
        # init adjacency matrix
        self.adjacency_matrix = np.zeros((len(nodes), len(nodes)))

        # create adjacency_matrix by edges
        for e in edges:
            self.adjacency_matrix[e[0], e[1]] = 1
            self.adjacency_matrix[e[1], e[0]] = 1

    def _get_connections(self):
        # make graph
        graph = nx.from_numpy_matrix(self.adjacency_matrix)
        # graph connections
        return sorted(nx.connected_components(graph), key=len, reverse=True)

    def _get_diameters(self, connections):
        # initialize diameter list
        diameters = np.zeros(len(connections))
        # append cluster index list to arange_list per row
        for i, c in enumerate(connections):
            data_index = np.array(list(c))
            subcluster = self.adjacency_matrix[data_index][:, data_index]
            g = nx.from_numpy_matrix(subcluster)
            diameters[i] = nx.diameter(g)
        return diameters

    def _get_row_arrangement(self, connections, diameters, max_diameter):
        # arrangement of clusters
        arrangement = []
        # tmp data
        row_diameter_sum = 0
        row_cluster_index = []
        # append cluster index list to arange_list per row
        for i, c in enumerate(connections):
            row_cluster_index.append(i)
            row_diameter_sum += diameters[i]

            if row_diameter_sum >= max_diameter:
                arrangement.append(row_cluster_index)
                row_diameter_sum = 0
                row_cluster_index = []

        if len(row_cluster_index) > 0:
            arrangement.append(row_cluster_index)
        return arrangement

    def _calc_coordinate(self, nodes, connections, diameters, max_diameter, arrangement, row_max_diameters):
        # calc coordinate
        pos = {}
        accum_height = 0
        total_heights = sum(row_max_diameters)
        total_width = max_diameter
        for i, l in enumerate(arrangement):
            accum_width = 0

            # calc coordinate per clusters
            for cluster_index in l:
                data_index = np.array(list(connections[cluster_index]))
                cluster_size = diameters[cluster_index] if diameters[cluster_index] > 0 else 1.

                subcluster = self.adjacency_matrix[data_index][:, data_index]
                g = nx.from_numpy_matrix(subcluster)
                if cluster_size > 3:
                    init_pos = {}
                    for j, n in enumerate(nodes[data_index]):
                        init_pos.update({j: [n[0], n[1]]})
                    k = 0.6 / cluster_size
                    p = nx.spring_layout(g, pos=init_pos, k=k)
                else:
                    p = nx.spectral_layout(g)

                # calculate plot coordinate
                tmp_width = cluster_size / total_width
                tmp_height = row_max_diameters[i] / total_heights
                for k in p.keys():
                    base = [accum_width * 1.5, accum_height * 1.5]
                    pos[data_index[k]] = p[k] * [tmp_width, tmp_height] + base
                accum_width += tmp_width
            accum_height += tmp_height
        return pos

    def _get_position(self, nodes, edges):
        # initialize adjacency_matrix
        self._init_adjacency_matrix(nodes, edges)
        # calculate connection from adjacency_matrix
        connections = self._get_connections()
        # calculate diameter list of clusters
        diameters = self._get_diameters(connections)

        # max diameter of cluster
        max_diameter = max(diameters)

        # arrangement of clusters
        arrangement = self._get_row_arrangement(connections, diameters, max_diameter)

        # calc row max List
        row_max_diameters = [max(diameters[l]) for l in arrangement]

        # calc coordinate
        pos = self._calc_coordinate(nodes, connections, diameters,
                                    max_diameter, arrangement,
                                    row_max_diameters)
        return pos

    def _plot(self, nodes, edges, node_sizes, colors):
        # calculate node position
        pos = self._get_position(nodes, edges)

        # make graph plot with self.adjacency_matrix
        plt.figure(figsize=self.fig_size)
        g = nx.from_numpy_matrix(self.adjacency_matrix)
        nx.draw_networkx(g, pos=pos,
                         node_size=node_sizes * self.node_size,
                         node_color=colors,
                         width=self.edge_width,
                         edge_color=[colors[e[0]] for e in edges],
                         with_labels=False)
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
            SpectralPresenter(fig_size, node_size, edge_width)]

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
