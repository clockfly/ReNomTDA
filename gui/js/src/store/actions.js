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

    let fd = new FormData();
    fd.append("file_id", context.state.file_id);
    fd.append("target_index", context.state.target_index);
    fd.append("algorithm", context.state.topologies[payload.index].algorithm);
    fd.append("mode", context.state.topologies[payload.index].mode);
    fd.append("clustering_algorithm", context.state.topologies[payload.index].clustering_algorithm);
    fd.append("train_size", context.state.topologies[payload.index].train_size);
    fd.append("k", context.state.topologies[payload.index].k);
    fd.append("eps", context.state.topologies[payload.index].eps);
    fd.append("min_samples", context.state.topologies[payload.index].min_samples);
    fd.append("resolution", context.state.topologies[payload.index].resolution);
    fd.append("overlap", context.state.topologies[payload.index].overlap);
    fd.append("color_index", context.state.topologies[payload.index].color_index);

    axios.post("/api/create", fd)
      .then(function(response) {
        context.commit('set_create_result', {
          'index': payload.index,
          'target_index': context.state.target_index,
          'hypercubes': response.data.hypercubes,
          'nodes': response.data.nodes,
          'edges': response.data.edges,
          'node_sizes': response.data.node_sizes,
          'colors': response.data.colors,
          'train_index': response.data.train_index,
        });
        context.commit('set_loading', {'loading': false});
      });
  }
}
