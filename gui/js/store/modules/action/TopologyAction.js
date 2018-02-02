import axios from 'axios'

// TODO
// actionのメソッドがreturnしないように変更する.
let TopologyAction = {
  load_file_list(context, payload) {
    context.commit('set_loading');
    return axios.get('/api/load_file_list')
      .then(function(response){
        context.commit("set_file_list", {
          "files": response.data.files,
        });
        context.commit('set_loading');
      });
  },

  load_file(context, payload) {
    context.commit('set_loading');
    context.commit('reset_data_histogram');

    let fd = new FormData();
    fd.append("file_id", payload.file_id);

    return axios.post('/api/load_file', fd)
      .then(function(response){
        let error = response.data.error;
        if (error == "LF0001"){
          alert("File not found. Please try again.");
          context.commit('set_loading');
          return
        }else if(error == "LF0002") {
          alert("File is too learge. Max file size is 10,000 rows.");
          context.commit('set_loading');
          return
        }

        context.commit('set_load_file_result', {
          'file_id': payload.file_id,
          'categorical_data_labels': response.data.categorical_data_labels,
          'numerical_data_labels': response.data.numerical_data_labels,
          'numerical_data': response.data.numerical_data,
          'numerical_data_mins': response.data.numerical_data_mins,
          'numerical_data_maxs': response.data.numerical_data_maxs,
          'numerical_data_means': response.data.numerical_data_means
        });
        context.commit('set_loading');
      });
  },

  create_topology(context, payload) {
    context.commit('set_loading');
    context.commit('reset_topology');

    let fd = new FormData();
    fd.append("rand_str", localStorage.getItem("rand_str"));
    fd.append("file_id", context.state.file_id);
    fd.append("create_topology_index", context.state.create_topology_index.toString());
    fd.append("colorize_topology_index", context.state.colorize_topology_index.toString());

    fd.append("columns", context.state.layout_columns);
    fd.append("mode", context.state.mode);
    fd.append("algorithm", context.state.algorithm_index);
    fd.append("resolution", context.state.resolution);
    fd.append("overlap", context.state.overlap);
    fd.append("eps", context.state.eps);
    fd.append("min_samples", context.state.min_samples);
    fd.append("color_index", context.state.color_index);
    fd.append("clustering_index", context.state.clustering_index);
    fd.append("class_count", context.state.class_count);
    fd.append("train_size", context.state.train_size);
    fd.append("neighbors", context.state.neighbors);

    fd.append("mode_col2", context.state.mode_col2);
    fd.append("algorithm_col2", context.state.algorithm_index_col2);
    fd.append("resolution_col2", context.state.resolution_col2);
    fd.append("overlap_col2", context.state.overlap_col2);
    fd.append("eps_col2", context.state.eps_col2);
    fd.append("min_samples_col2", context.state.min_samples_col2);
    fd.append("color_index_col2", context.state.color_index_col2);
    fd.append("clustering_index_col2", context.state.clustering_index_col2);
    fd.append("class_count_col2", context.state.class_count_col2);
    fd.append("train_size_col2", context.state.train_size_col2);
    fd.append("neighbors_col2", context.state.neighbors_col2);

    return axios.post('/api/create', fd)
      .then(function(response){
        let error = response.data.error;
        if (error){
          alert("Can't create node, please change parameters.");
          context.commit('set_loading');
          return
        }

        context.commit('set_create_topology_result', {
          'nodes': response.data.canvas_data[0].nodes,
          'edges': response.data.canvas_data[0].edges,
          'colors': response.data.canvas_data[0].colors,
          'sizes': response.data.canvas_data[0].sizes,
          'train_index': response.data.canvas_data[0].train_index,
          'nodes_col2': response.data.canvas_data[1].nodes,
          'edges_col2': response.data.canvas_data[1].edges,
          'colors_col2': response.data.canvas_data[1].colors,
          'sizes_col2': response.data.canvas_data[1].sizes,
          'train_index_col2': response.data.canvas_data[1].train_index,
          'data_color_values': response.data.histogram_data[0].data_color_values,
          'data_color_values_col2': response.data.histogram_data[1].data_color_values,
          "statistic_value": response.data.histogram_data[0].statistic_value,
          "statistic_value_col2": response.data.histogram_data[1].statistic_value,
          "pca_result": response.data.pca_result
        });
        context.commit('set_loading');
      });
  },

  search_topology(context, payload) {
    context.commit('set_loading');
    context.commit('reset_colors');

    let fd = new FormData();
    fd.append("rand_str", localStorage.getItem("rand_str"));
    fd.append("file_id", context.state.file_id);
    fd.append("create_topology_index", context.state.create_topology_index.toString());
    fd.append("colorize_topology_index", context.state.colorize_topology_index.toString());

    fd.append("is_categorical", +payload.is_categorical);
    fd.append("search_column_index", payload.search_column_index);
    fd.append("operator_index", payload.operator_index);
    fd.append("search_value", payload.search_txt);
    fd.append("color_index", context.state.color_index);
    fd.append("mode", context.state.mode);
    fd.append("color_index_col2", context.state.color_index_col2);
    fd.append("mode_col2", context.state.mode_col2);

    return axios.post('/api/search', fd)
      .then(function(response){
        context.commit('set_search_result', {
          'colors': response.data.canvas_colors[0].colors,
          'colors_col2': response.data.canvas_colors[1].colors
        });
        context.commit('set_loading');
      });
  },

  click_node(context, payload) {
    let fd = new FormData();
    fd.append("rand_str", localStorage.getItem("rand_str"));
    fd.append("file_id", context.state.file_id);
    fd.append("clicknode", payload.click_node_index);
    fd.append("columns", payload.columns);
    if (payload.columns == 0){
      fd.append("mode", context.state.mode);
    }else if(payload.columns == 1){
      fd.append("mode", context.state.mode_col2);
    }

    return axios.post('/api/click', fd)
    .then(function(response){
      context.commit('set_click_node_result', {
        'click_node_index': payload.click_node_index,
        'node_categorical_data': response.data.categorical_data,
        'node_data': response.data.data
      })
    });
  },

  reset(context) {
    context.commit('set_loading');
    context.commit('reset_topology');

    return axios.get('/api/reset')
      .then(function(response){
        context.commit('set_loading');
      });
  },
  reset_all(context) {
    context.commit('set_loading');
    context.commit('reset_all');

    return axios.get('/api/reset')
      .then(function(response){
        context.commit('set_loading');
      });
  },
  set_random(context) {
    let fd = new FormData();
    fd.append("rand_str", localStorage.getItem("rand_str"));

    return axios.post('/api/set_random')
      .then(function(response) {
        localStorage.setItem("rand_str", response.data.rand_str);
      })
  }
}

export default TopologyAction;