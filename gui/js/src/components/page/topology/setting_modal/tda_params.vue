<template>
  <div id="tda-params">
    <div class="mapping-params">Mapping Params</div>
    <mapping-params :columnIndex="columnIndex"></mapping-params>

    <div class="clustering-params">Clustering Params</div>
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
  .mapping-params {
    margin-top: 16px;
  }
  .clustering-params {
    margin-top: 16px;
  }
}
</style>
