<template>
  <div id="dbscan">
    <div class="input-group vertical">
      <label for="eps">Epsilon: {{ eps }}</label>
      <input type="range" min="0.01" max="1" step="0.01" class="slider" v-model="eps">
    </div>

    <div class="input-group vertical">
      <label for="min_samples">min_samples: {{ min_samples }}</label>
      <input type="range" min="1" max="5" step="1" class="slider" v-model="min_samples">
    </div>
  </div>
</template>

<script>
export default {
  name: "Dbscan",
  props: {
    "columnIndex": {
      type: Number,
      require: true,
    }
  },
  computed: {
    "eps": {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].eps;
      },
      set: function(val) {
        this.$store.commit("set_eps", {
          "index": this.columnIndex,
          "val": val,
        })
      },
    },
    "min_samples": {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].min_samples;
      },
      set: function(val) {
        this.$store.commit("set_min_samples", {
          "index": this.columnIndex,
          "val": val,
        })
      },
    }
  }
}
</script>

<style lang="scss" scoped>
#dbscan {
}
</style>
