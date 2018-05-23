export default {
  click_node_data: function(state) {
    let click_node_data = []
    for(let i of state.click_node_data_ids) {
      click_node_data.push(state.file_data[i]);
    }
    return click_node_data;
  }
}
