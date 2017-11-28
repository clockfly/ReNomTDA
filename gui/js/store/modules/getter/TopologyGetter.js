let TopologyGetter = {
  loading_state: function(state) {
    return state.loading;
  },
  layout_columns: function(state) {
    return state.layout_columns;
  },
  color_labels: function(state){
    let ret_array = new Array();
    for(let i=0; i < state.colorize_topology_index.length; i++){
      ret_array.push(state.numerical_data_labels[state.colorize_topology_index[i]])
    }
    return ret_array
  }
}

export default TopologyGetter;