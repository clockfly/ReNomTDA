export default {
  click_node_data: function(state) {
    let click_node_data = []
    for(let i of state.click_node_data_ids) {
      console.log(i);
      console.log(state.number_data[i]);
      click_node_data.push(state.number_data[i]);
    }
    return click_node_data;
  }
}
