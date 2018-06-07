<template>
  <div id='file_summary'>
    <span class='summary_title'>File Summary</span>
    <span class='summary_value margin-left-12'>Column {{ columns }}</span>
    <span class='summary_value margin-left-12'>Row {{ row }}</span>
  </div>
</template>

<script>
import {mapState} from 'vuex'

export default {
  name: 'FileSummary',
  computed: mapState(['row', 'columns']),
}
</script>

<style lang='scss' scoped>
#file_summary {
  font-size: 12px;
  line-height: 26px;

  .summary_value {
    padding-left: 12px;
    border-left: 1px solid #ccc;
  }
}
</style>
