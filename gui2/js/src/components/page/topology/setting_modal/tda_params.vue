<template>
  <div id="tda-params">
    <mapping-params :columnIndex="columnIndex"></mapping-params>
    <dbscan :columnIndex="columnIndex"></dbscan>
  </div>
</template>

<script>
import Dbscan from './dbscan.vue'
import MappingParams from './mapping_params.vue'

export default {
  name: "TdaParams",
  components: {
    "mapping-params": MappingParams,
    "dbscan": Dbscan,
  },
  props: {
    "columnIndex": {
      type: Number,
      require: true,
    }
  },
}
</script>

<style lang="scss" scoped>
#tda-params {
}
</style>
