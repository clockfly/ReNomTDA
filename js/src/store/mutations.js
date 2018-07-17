export default {
  /*
  data selection page
  */
  set_file_list: function(state, payload) {
    state.files = payload.files;
  },
  set_file_id: function(state, payload) {
    state.file_id = payload.file_id;
  },
  set_load_result: function(state, payload) {
    state.row = payload.data.row;
    state.columns = payload.data.columns;
    state.data_header = payload.data.data_header;
    state.file_data = payload.data.file_data;

    state.number_index = payload.data.number_index;
    state.number_columns = payload.data.number_columns;
    state.text_columns = payload.data.text_columns;

    state.number_data = payload.data.number_data;
    state.text_data = payload.data.text_data;

    state.hist_data = payload.data.hist_data;
    state.topo_hist = payload.data.topo_hist;

    state.data_mean = payload.data.data_mean;
    state.data_var = payload.data.data_var;
    state.data_std = payload.data.data_std;
    state.data_min = payload.data.data_min;
    state.data_25percentile = payload.data.data_25percentile;
    state.data_50percentile = payload.data.data_50percentile;
    state.data_75percentile = payload.data.data_75percentile;
    state.data_max = payload.data.data_max;
  },
  set_loading: function(state, payload) {
    state.loading.splice(payload.index, 1, payload.loading);
  },
  set_select_column_name: function(state, payload) {
    state.selected_column = payload.name;
  },
  set_target_index: function(state, payload) {
    state.target_index = payload.target_index;
    state.topologies[0].color_index = payload.target_index;
    state.topologies[1].color_index = payload.target_index;
  },

  /*
  topology page
  */
  set_setting_modal: function(state, payload) {
    state.show_setting_modal = payload.is_show;
  },
  set_search_modal: function(state, payload) {
    state.show_search_modal = payload.is_show;
  },
  set_algorithm: function(state, payload) {
    state.topologies[payload.index].algorithm = payload.val;
  },
  set_clustering_algorithm: function(state, payload) {
    state.topologies[payload.index].clustering_algorithm = payload.val;
  },
  set_mode: function(state, payload) {
    state.topologies[payload.index].mode = payload.val;
  },
  set_k: function(state, payload) {
    state.topologies[payload.index].k = payload.val;
  },
  set_eps: function(state, payload) {
    state.topologies[payload.index].eps = payload.val;
  },
  set_min_samples: function(state, payload) {
    state.topologies[payload.index].min_samples = payload.val;
  },
  set_train_size: function(state, payload) {
    state.topologies[payload.index].train_size = payload.val;
  },
  set_resolution: function(state, payload) {
    state.topologies[payload.index].resolution = payload.val;
  },
  set_overlap: function(state, payload) {
    state.topologies[payload.index].overlap = payload.val;
  },
  set_reduction_result: function(state, payload) {
    state.topologies[payload.index].point_cloud = payload.point_cloud;
  },
  set_create_result: function(state, payload) {
    state.topologies[payload.index].hypercubes = payload.hypercubes;
    state.topologies[payload.index].nodes = payload.nodes;
    state.topologies[payload.index].edges = payload.edges;
    state.topologies[payload.index].node_sizes = payload.node_sizes;
    state.topologies[payload.index].colors = payload.colors;
    state.topologies[payload.index].train_index = payload.train_index;
  },
  set_color_index: function(state, payload) {
    state.topologies[payload.index].color_index = payload.val;
  },
  reset_topology: function(state, payload) {
    state.click_node_data_ids = [];
    for(let t of state.topologies) {
      t.nodes = [];
      t.edges = [];
    }
  },
  set_click_node_ids: function(state, payload) {
    state.click_node_ids[payload.index].push(payload.click_node_index);
  },
  remove_click_node_ids: function(state, payload) {
    let i = state.click_node_ids[payload.index].indexOf(payload.click_node_index);
    state.click_node_ids[payload.index].splice(i, 1);
  },
  set_click_node: function(state, payload) {
    const ids = state.topologies[payload.index].hypercubes[payload.click_node_index];
    let a = new Set(state.click_node_data_ids);
    let b = new Set(ids);
    let union = new Set([...a, ...b]);
    state.click_node_data_ids = [...union];
  },
  remove_click_node: function(state, payload) {
    const ids = state.topologies[payload.index].hypercubes[payload.click_node_index];
    let a = new Set(state.click_node_data_ids);
    let b = new Set(ids);
    let diff = new Set([...a].filter(x => !b.has(x)));
    state.click_node_data_ids = [...diff];
  },
  set_search_color: function(state, payload) {
    state.topologies[payload.index].colors = payload.colors;
  },
  set_search_type: function(state, payload) {
    state.search_type = payload.val;
  },
  set_visualize_mode: function(state, payload) {
    state.visualize_mode = payload.val;
  }
}
