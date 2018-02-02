let TopologyMutation = {
  reset_data_histogram(state) {
    state.numerical_data = undefined;
  },
  reset_all(state) {
    state.layout_columns= 0;
    state.algorithm_index = 0;
    state.algorithm_index_col2 = 0;

    state.mode = 0;
    state.resolution = 25;
    state.overlap = 0.5;
    state.eps = 0.5;
    state.min_samples = 3;
    state.color_index = 0;
    state.clustering_index = 0;
    state.class_count = 3;

    state.mode_col2 = 0;
    state.resolution_col2 = 25;
    state.overlap_col2 = 0.5;
    state.eps_col2 = 0.5;
    state.min_samples_col2 = 3;
    state.color_index_col2 = 0;
    state.clustering_index_col2 = 0;
    state.class_count_col2 = 3;

    state.pca_result = undefined;

    state.show_histogram = false;
    state.show_spring = false;

    state.search_txt = "";

    state.nodes = undefined;
    state.edges = undefined;
    state.colors = undefined;
    state.sizes = undefined;

    state.nodes_col2 = undefined;
    state.edges_col2 = undefined;
    state.colors_col2 = undefined;
    state.sizes_col2 = undefined;

    state.click_node_index = undefined;
    state.node_categorical_data = undefined;
    state.node_data = undefined;
    state.show_category = 0;
  },
  reset_topology(state) {
    state.nodes = undefined;
    state.edges = undefined;
    state.colors = undefined;
    state.sizes = undefined;
    state.nodes_col2 = undefined;
    state.edges_col2 = undefined;
    state.colors_col2 = undefined;
    state.sizes_col2 = undefined;
    state.click_node_index = undefined;
    state.node_categorical_data = undefined;
    state.node_data = undefined;
    state.show_category = 0;
    state.pca_result = undefined;
  },
  reset_colors(state) {
    state.colors = undefined;
    state.colors_col2 = undefined;
  },

  set_loading(state) {
    state.loading = !state.loading;
  },
  set_show_histogram(state) {
    state.show_histogram = !state.show_histogram
  },
  set_show_spring(state) {
    state.show_spring = !state.show_spring
  },

  // data selection page
  set_file_id(state, payload) {
    state.file_id = payload.file_id;
  },
  set_file_list(state, payload) {
    state.files = payload.files;
  },
  set_selected_index(state, payload) {
    state.create_topology_index = payload.create_topology_index;
    state.colorize_topology_index = payload.colorize_topology_index;
  },
  set_load_file_result(state, payload) {
    state.categorical_data_labels = payload.categorical_data_labels;
    state.numerical_data_labels = payload.numerical_data_labels;
    state.numerical_data = payload.numerical_data;
    state.numerical_data_mins = payload.numerical_data_mins;
    state.numerical_data_maxs = payload.numerical_data_maxs;
    state.numerical_data_means = payload.numerical_data_means;
  },

  // layout selector
  set_layout(state, payload) {
    state.layout_columns = payload.layout_columns;
  },

  // dimension reduction selector
  set_dimension_reduction_algorithm(state, payload) {
    if (payload.column == 0){
      state.algorithm_index = payload.algorithm_index;
    }else if(payload.column == 1){
      state.algorithm_index_col2 = payload.algorithm_index;
    }
  },

  // param selector
  set_mode(state, payload) {
    if (payload.column == 0){
      state.mode = payload.mode;
    }else if(payload.column == 1){
      state.mode_col2 = payload.mode;
    }
  },
  set_resolution(state, payload) {
    if (payload.column == 0){
      state.resolution = payload.resolution;
    }else if(payload.column == 1){
      state.resolution_col2 = payload.resolution;
    }
  },
  set_overlap(state, payload) {
    if (payload.column == 0){
      state.overlap = payload.overlap;
    }else if(payload.column == 1){
      state.overlap_col2 = payload.overlap;
    }
  },
  set_epsilon(state, payload) {
    if (payload.column == 0){
      state.eps = payload.eps;
    }else if(payload.column == 1){
      state.eps_col2 = payload.eps;
    }
  },
  set_min_samples(state, payload) {
    if (payload.column == 0){
      state.min_samples = payload.min_samples;
    }else if(payload.column == 1){
      state.min_samples_col2 = payload.min_samples;
    }
  },
  set_color_index(state, payload) {
    if (payload.column == 0){
      state.color_index = payload.color_index;
    }else if(payload.column == 1){
      state.color_index_col2 = payload.color_index;
    }
  },
  set_clustering_index(state, payload) {
    if (payload.column == 0){
      state.clustering_index = payload.clustering_index;
    }else if(payload.column == 1){
      state.clustering_index_col2 = payload.clustering_index;
    }
  },
  set_class_count(state, payload) {
    if (payload.column == 0){
      state.class_count = payload.class_count;
    }else if(payload.column == 1){
      state.class_count_col2 = payload.class_count;
    }
  },
  set_train_size(state, payload) {
    if (payload.column == 0){
      state.train_size = payload.train_size;
    }else if(payload.column == 1){
      state.train_size_col2 = payload.train_size;
    }
  },
  set_neighbors(state, payload) {
    if (payload.column == 0){
      state.neighbors = payload.neighbors;
    }else if(payload.column == 1){
      state.neighbors_col2 = payload.neighbors;
    }
  },

  set_create_topology_result(state, payload) {
    state.nodes = payload.nodes;
    state.edges = payload.edges;
    state.colors = payload.colors;
    state.sizes = payload.sizes;
    state.train_index = payload.train_index;
    state.nodes_col2 = payload.nodes_col2;
    state.edges_col2 = payload.edges_col2;
    state.colors_col2 = payload.colors_col2;
    state.sizes_col2 = payload.sizes_col2;
    state.train_index_col2 = payload.train_index_col2;
    state.data_color_values = payload.data_color_values;
    state.data_color_values_col2 = payload.data_color_values_col2;
    state.statistic_value = payload.statistic_value;
    state.statistic_value_col2 = payload.statistic_value_col2;
    state.pca_result = payload.pca_result;
  },

  set_search_result(state, payload) {
    state.colors = payload.colors;
    state.colors_col2 = payload.colors_col2;
  },

  set_click_node_result(state, payload) {
    state.click_node_index = payload.click_node_index;
    state.node_categorical_data = payload.node_categorical_data;
    state.node_data = payload.node_data;
  },
}

export default TopologyMutation;