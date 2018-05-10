import axios from 'axios'

export default {
  /*
  data selection page
  */
  load_files(context, payload) {
    context.commit('set_loading', {'loading': true});
    return axios.get('/api/files')
      .then(function(response){
        if(response.data.error_msg) {
          alert(response.data.error_msg);
          context.commit('set_loading', {'loading': false});
          return;
        }
        context.commit('set_file_list', {
          'files': response.data.files,
        });
        context.commit('set_loading', {'loading': false});
      });
  },
  load_file(context, payload) {
    context.commit('set_loading', {'loading': true});
    const url = '/api/files/' + context.state.file_id
    axios.get(url)
      .then(function(response){
        if(response.data.error_msg) {
          alert(response.data.error_msg);
          context.commit('set_loading', {'loading': false});
          return;
        }

        context.commit('set_load_result', {
          'data': response.data,
        });
        context.commit('set_loading', {'loading': false});
      });
  },

  /*
  topology page
  */
  async create(context, payload) {
    context.commit('set_loading', {'loading': true});
    context.commit('set_setting_modal', {"is_show": false});

    context.commit('reset_topology');

    let query = {
      "file_id": context.state.file_id,
      "target_index": context.state.target_index,
      "algorithm": context.state.topologies[payload.index].algorithm,
      "mode": context.state.topologies[payload.index].mode,
      "clustering_algorithm": context.state.topologies[payload.index].clustering_algorithm,
      "train_size": context.state.topologies[payload.index].train_size,
      "k": context.state.topologies[payload.index].k,
      "eps": context.state.topologies[payload.index].eps,
      "min_samples": context.state.topologies[payload.index].min_samples,
      "resolution": context.state.topologies[payload.index].resolution,
      "overlap": context.state.topologies[payload.index].overlap,
      "color_index": context.state.topologies[payload.index].color_index,
    }

    axios.get("/api/create", {
      params: query
    }).then(function(response) {
        context.commit('set_create_result', {
          'index': payload.index,
          'target_index': context.state.target_index,
          'hypercubes': response.data.hypercubes,
          'point_cloud': response.data.point_cloud,
          'nodes': response.data.nodes,
          'edges': response.data.edges,
          'node_sizes': response.data.node_sizes,
          'colors': response.data.colors,
          'train_index': response.data.train_index,
        });
        context.commit('set_loading', {'loading': false});
      });
  },
  export_data(context, payload) {
    context.commit('set_loading', {'loading': true});

    let fd = new FormData();
    fd.append('out_file_name', payload.out_file_name);
    fd.append("file_id", context.state.file_id);
    fd.append('click_node_data_ids', JSON.stringify(context.state.click_node_data_ids));

    axios.post('/api/export', fd)
      .then(function(response) {
      });

    context.commit('set_loading', {'loading': false});
  },
  search(context, payload) {
    context.commit('set_loading', {'loading': true});

    let fd = new FormData();
    let dict = {
      "file_id": context.state.file_id,
      'target_index': context.state.target_index,
      'mode': context.state.topologies[payload.index].mode,
      'clustering_algorithm': context.state.topologies[payload.index].clustering_algorithm,
      "train_size": context.state.topologies[payload.index].train_size,
      "k": context.state.topologies[payload.index].k,
      "eps": context.state.topologies[payload.index].eps,
      "min_samples": context.state.topologies[payload.index].min_samples,
      'point_cloud': context.state.topologies[payload.index].point_cloud,
      'hypercubes': context.state.topologies[payload.index].hypercubes,
      'nodes': context.state.topologies[payload.index].nodes,
      'edges': context.state.topologies[payload.index].edges,
      'colors': context.state.topologies[payload.index].colors,
      'search_type': payload.search_type,
      'search_conditions': payload.conditions
    }
    fd.append('data', JSON.stringify(dict));

    axios.post('/api/search', fd)
      .then(function(response) {
        context.commit('set_search_color', {
          'index': payload.index,
          'colors': response.data.colors,
        });
      });

    context.commit("set_search_modal", {"is_show": false});
    context.commit('set_loading', {'loading': false});
  }
}
