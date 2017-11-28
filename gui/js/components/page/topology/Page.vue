<template>
  <div id="page" class="row">
    <leftsidebar class="col-md-2 col-lg-2"></leftsidebar>
    <center class="col-md-8 col-lg-8"></center>
    <rightsidebar class="col-md-2 col-lg-2"></rightsidebar>
  </div>
</template>

<script>
import LeftSideBar from './leftsidebar/LeftSideBar.vue'
import Center from './center/Center.vue'
import RightSideBar from './rightsidebar/RightSideBar.vue'

export default {
  name: "TopologyPage",
  components: {
    'leftsidebar': LeftSideBar,
    'center': Center,
    'rightsidebar': RightSideBar
  },
  created: function() {
    if (this.$store.state.topology.filename === "") {
      this.$router.push({ path: '/' });
    }
  }
}
</script>

<style lang="scss">
#page {
  $header_height: 50px;

  height: calc(100vh - #{$header_height});
}
</style>