<template>
  <div id="select-table">
    <div class="table-area">
      <table class="node-data-table">
        <thead class="table-row">
          <tr>
            <th v-for="(l, index) in $store.state.data_header" :key="index" class="table-header">{{ l }}</th>
          </tr>
        </thead>
        <tbody class="table-row">
          <tr v-for="(row, index) in $store.getters.click_node_data"
            :key="index">
            <td v-for="(d, index) in row" :key="index" class="table-data">
              {{ d }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="export-button-area">
      <button class="export-button" @click="exportData">
        <i class="fa fa-pencil-square-o" aria-hidden="true"></i> export
      </button>
    </div>
  </div>
</template>

<script>

export default {
  name: "SelectTable",
  methods: {
    exportData: function() {
      let out_file_name = window.prompt('ファイル名を入力してください。', '');
      if(out_file_name && out_file_name.length > 0){
        this.$store.dispatch('export_data', {
          'out_file_name': out_file_name,
        });
      }
    }
  }
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

  .table-area {
    width: 95%;
    height: 80%;
    margin: 0 auto;
    overflow: auto;
    .table-header {
      height: $table_item_height;
      margin: 0;
      padding: $table_item_padding;
      line-height: $table_item_height;
      font-size: $table_font_size;
      text-align: center;
    }
    .table-data {
      height: $table_item_height;
      margin: 0;
      padding: $table_item_padding;
      line-height: $table_item_height;
      font-size: $table_font_size;
      text-align: center;
    }
  }

  .export-button-area {
    display: flex;
    height: 10%;

    .export-button {
      margin-left: auto;
      margin-right: 48px;
    }
  }
}
</style>
