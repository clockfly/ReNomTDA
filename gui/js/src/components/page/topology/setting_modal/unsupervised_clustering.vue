<template>
  <div id="unsupervised-clustering">
    <div class="input-group vertical">
      <label for="mode">Unsupervised Clustering</label>
      <select v-model="algorithm">
        <option v-for="(v, index) in clusterings"
          :value="index">{{v}}</option>
      </select>
    </div>

    <component :is="clusterings[algorithm]" :columnIndex="columnIndex"></component>
  </div>
</template>

<script>
import KMeans from './kmeans.vue'
import Dbscan from './dbscan.vue'

export default {
  name: "UnsupervisedClustering",
  components: {
    "K-Means": KMeans,
    "DBSCAN": Dbscan,
  },
  props: {
    "columnIndex": {
      type: Number,
      require: true,
    }
  },

  computed: {
    clusterings() {
      return this.$store.state.unsupervised_clusterings;
    },
    algorithm: {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].clustering_algorithm;
      },
      set: function(val) {
        this.$store.commit("set_clustering_algorithm", {
          "index": this.columnIndex,
          "val": val,
        })
      }
    }
  }
}
</script>

<style lang="scss" scoped>
#unsupervised-clustering {
}
</style>
