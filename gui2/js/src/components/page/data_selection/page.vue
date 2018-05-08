<template>
  <div id='data_selection' class='row'>
    <center class='col-md-12 col-lg-12'></center>
  </div>
</template>

<script>
import Center from './center/center.vue'

export default {
  name: 'DataSelectionPage',
  components: {
    'center': Center
  },
  created: function() {
    this.$store.dispatch('load_files');
  }
}
</script>

<style lang='scss' scoped>
#data_selection {
  $header-height: 44px;

  height: calc(100vh - #{$header_height});
  margin-top: $header-height;
}
</style>
