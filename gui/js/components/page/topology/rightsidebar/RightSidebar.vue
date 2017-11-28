<template>
  <div id="right_sidebar">
    <search-box></search-box>
    <node-data-area></node-data-area>
  </div>
</template>

<script>
import SearchBox from './SearchBox.vue'
import NodeDataArea from './NodeDataArea.vue'

export default {
  name: "TopologyRightSideBar",
  components: {
    'search-box': SearchBox,
    'node-data-area': NodeDataArea
  },
  data: function() {
    return {
      advanced_search: false
    }
  }
}
</script>

<style lang="scss">
#right_sidebar {
  margin: 0;
  padding: 0;
  background-color: #f5f5f5;
}
</style>