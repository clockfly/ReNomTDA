<template>
  <div id="left_sidebar">
    <fileloader></fileloader>
  </div>
</template>

<script>
import FileLoader from './FileLoader.vue'

export default {
  name: "DataSelectionLeftSideBar",
  components: {
    'fileloader': FileLoader
  }
}
</script>

<style lang="scss">
#left_sidebar {
  margin: 0;
  padding: 0;
  background-color: #f5f5f5;
}
</style>