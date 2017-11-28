<template>
  <div id="nodedata_area">
    <div class="node_data_count" v-if="node_data">
      <div v-if="node_data.length > 1">
        node has {{ node_data.length }} data.
      </div>
    </div>

    <node-data v-for="(data, index) in node_data" :data="data" :index="index" :key="index" :categoricaldata="categorical_data[index]"></node-data>
  </div>
</template>

<script>
import NodeData from './NodeData.vue'

export default {
  name: "TopologyNodeDataArea",
  components: {
    'node-data': NodeData
  },
  computed: {
    node_data() {
      return this.$store.state.topology.node_data;
    },
    categorical_data() {
      return this.$store.state.topology.node_categorical_data
    },
  }
}
</script>

<style lang="scss">
#nodedata_area {
  height: 60%;
  overflow: scroll;
  .node_data_count {
    margin: 10px 10%;
  }
}
</style>