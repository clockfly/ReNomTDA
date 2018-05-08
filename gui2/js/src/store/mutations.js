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
    state.number_index = payload.data.number_index;
    state.hist_data = payload.data.hist_data;

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
    state.loading = payload.loading;
  },
  set_select_column_name: function(state, payload) {
    state.selected_column = payload.name;
  },
  set_target_name: function(state, payload) {
    state.target_index = payload.target_index;
  },

  /*
  topology page
  */
  set_setting_modal: function(state, payload) {
    state.show_setting_modal = payload.is_show;
  }
}
