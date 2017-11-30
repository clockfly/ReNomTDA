<template>
  <div id="dimension_reduction_selector">
    <div class="input-group vertical">
      <label for="algorithm">Dimension Reduction</label>
      <select v-model="algorithm" v-on:change="change_dimension_reduction">
        <option v-for="(a, index) in algorithms" :value="index">{{a}}</option>
      </select>
    </div>
  </div>
</template>

<script>
export default {
  name: "DimensionReductionSelector",
  props: ["column"],
  data: function() {
    return {
      algorithms: ["PCA", "TSNE", "MDS", "Isomap", "Auto Encoder 2Layer", "Auto Encoder 3Layer", "Auto Encoder 4Layer", "None"],
      algorithm: 0
    }
  },
  methods: {
    change_dimension_reduction: function() {
      this.$store.commit('set_dimension_reduction_algorithm', {
        'algorithm_index': this.algorithm,
        'column': this.column
      });
    }
  }
}
</script>
