<template>
  <div id="supervised-clustering">
    <div class="input-group vertical">
      <label for="mode">Unsupervised Clustering</label>
      <select v-model="algorithm">
        <option v-for="(v, index) in clusterings" :value="index">{{v}}</option>
      </select>
    </div>

    <div class="input-group vertical">
      <label for="train_data_size">Train Size: {{ train_data_size }}</label>
      <input type="range" min="0.1" max="0.9" step="0.1" class="slider" v-model="train_data_size">
    </div>

    <k-nn v-if="algorithm == 0" :columnIndex="columnIndex"></k-nn>
  </div>
</template>

<script>
import KNN from './knn.vue'

export default {
  name: "SupervisedClustering",
  components: {
    "k-nn": KNN,
  },
  props: {
    "columnIndex": {
      type: Number,
      require: true,
    }
  },
  computed: {
    clusterings() {
      return this.$store.state.supervised_clusterings;
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
    },
    train_data_size: {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].train_size;
      },
      set: function(val) {
        this.$store.commit("set_train_size", {
          "index": this.columnIndex,
          "val": val,
        })
      }
    }
  }
}
</script>

<style lang="scss" scoped>
#supervised-clustering {
}
</style>
