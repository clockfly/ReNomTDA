<template>
  <div id="select-table">
    <div class="table_area">
      <table class="node_data_table">
        <thead class="table_row">
          <tr>
            <th v-for="(l, index) in $store.state.data_header" :key="index" class="table_header">{{ l }}</th>
          </tr>
        </thead>
        <tbody class="table_row">
          <tr v-for="(row, index) in $store.getters.click_node_data"
            :key="index">
            <td v-for="(d, index) in row" :key="index" class="table_data">
              {{ d }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script>

export default {
  name: "SelectTable",
}
</script>

<style lang="scss" scoped>
#select-table {
  $table_item_width: 50px;
  $table_item_height: 12px;
  $table_item_padding: 4px;
  $table_font_size: 9px;

  width: 100%;
  height: 100%;

  .table_area {
    width: 95%;
    height: 80%;
    margin: 0 auto;
    overflow: auto;
    .table_header {
      height: $table_item_height;
      margin: 0;
      padding: $table_item_padding;
      line-height: $table_item_height;
      font-size: $table_font_size;
      text-align: center;
    }
    .table_data {
      height: $table_item_height;
      margin: 0;
      padding: $table_item_padding;
      line-height: $table_item_height;
      font-size: $table_font_size;
      text-align: center;
    }
  }
}
</style>
