<template>
  <div class="common-params">
    <div class="modal-param">
      <div class="input-group vertical">
        <label for="algorithm">Dimension Reduction</label>
        <select v-model="algorithm">
          <option v-for="(v, index) in $store.state.algorithms"
            :value="index">{{v}}</option>
        </select>
      </div>

      <div class="input-group vertical">
        <label for="mode">Analysis Mode</label>
        <select v-model="mode">
          <option v-for="(v, index) in $store.state.modes"
            :value="index">{{v}}</option>
        </select>
      </div>

      <component :is="param_components[mode]" :columnIndex="columnIndex"></component>
    </div>
  </div>
</template>

<script>
import UnsupervisedClustering from './unsupervised_clustering.vue'
import SupervisedClustering from './supervised_clustering.vue'
import TdaParams from './tda_params.vue'

export default {
  name: "CommonParams",
  components: {
    "unsupervised-clustering": UnsupervisedClustering,
    "supervised-clustering": SupervisedClustering,
    "tda-params": TdaParams,
  },
  props: {
    "columnIndex": {
      type: Number,
      required: true
    }
  },
  data: function() {
    return {
      "param_components": ["", "unsupervised-clustering", "supervised-clustering", "tda-params"],
    }
  },
  computed: {
    algorithm: {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].algorithm;
      },
      set: function(val) {
        this.$store.commit("set_algorithm", {
          "index": this.columnIndex,
          "val": val
        });
      }
    },
    mode: {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].mode;
      },
      set: function(val) {
        this.$store.commit("set_mode", {
          "index": this.columnIndex,
          "val": val
        });
      }
    },
  },

}
</script>

<style lang="scss" scoped>
.common-params {
  display: flex;
  flex-direction: column;
}
</style>
