<template>
  <div id="data_selection" class="row">
    <leftsidebar class="col-md-2 col-lg-2 left_sidebar"></leftsidebar>
    <center class="col-md-10 col-lg-10"></center>
  </div>
</template>

<script>
import LeftSideBar from './LeftSideBar.vue'
import Center from './Center.vue'

export default {
  name: "DataSelectionPage",
  components: {
    'leftsidebar': LeftSideBar,
    'center': Center
  },
  created: function() {
    this.$store.dispatch("load_file_list");
  }
}
</script>

<style lang="scss">
#data_selection {
  $header_height: 50px;

  height: calc(100vh - #{$header_height});

  .left_sidebar {
    padding: 12px;
  }
}
</style>