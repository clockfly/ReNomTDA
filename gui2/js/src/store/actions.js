import axios from 'axios'

export default {
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
}
