<template>
  <div id='center'>
    <div class='file_summary_area margin-top-16'>
      <file-loader class='margin-left-48'></file-loader>
      <file-summary class='margin-left-48'></file-summary>
    </div>

    <div class='data_table_area margin-top-16'>
      <data-table></data-table>
    </div>

    <div class='data_export_area margin-top-16 margin-bottom-16'>
      <next-button class='margin-right-48'></next-button>
      <target-selector class='margin-right-48'></target-selector>
    </div>
  </div>
</template>

<script>
import DataTable from './data_table/data_table.vue'
import FileLoader from './file_loader.vue'
import FileSummary from './file_summary.vue'
import NextButton from './next_button.vue'
import TargetSelector from './target_selector.vue'

export default {
  name: 'DataSelectionCenter',
  components: {
    'data-table': DataTable,
    'file-loader': FileLoader,
    'file-summary': FileSummary,
    'next-button': NextButton,
    'target-selector': TargetSelector,
  }
}
</script>

<style lang="scss" scoped>
#center {
  $header-height: 44px;
  $input-line-height: 24px;

  width: 100%;
  max-width: 1280px;
  height: calc(100vh - #{$header-height});
  margin: 0 auto;
  padding: 0;


  .file_summary_area, .data_table_global_area, .data_export_area {
    display: -webkit-flex;
    display: flex;

    width: 100%;
    height: $input-line-height;
  }

  .data_export_area {
    -webkit-flex-direction: row-reverse;
    flex-direction: row-reverse;
  }

  .data_table_area {
    width: 100%;
    height: calc(100% - #{$input-line-height}*2 - 16px*5);
  }
}
</style>
