<template>
  <div id="mapping-params">
    <div class="input-group vertical">
      <label for="resolution">resolution: {{ resolution }}</label>
      <input type="range" min="5" max="50" step="5" class="slider" v-model="resolution">
    </div>

    <div class="input-group vertical">
      <label for="resolution">overlap: {{ overlap }}</label>
      <input type="range" min="0.1" max="2" step="0.1" class="slider" v-model="overlap">
    </div>
  </div>
</template>

<script>
export default {
  name: "MappingParams",
  props: {
    "columnIndex": {
      type: Number,
      require: true,
    }
  },
  computed: {
    "resolution": {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].resolution;
      },
      set: function(val) {
        this.$store.commit("set_resolution", {
          "index": this.columnIndex,
          "val": val,
        })
      },
    },
    "overlap": {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].overlap;
      },
      set: function(val) {
        this.$store.commit("set_overlap", {
          "index": this.columnIndex,
          "val": val,
        })
      },
    }
  }
}
</script>

<style lang="scss" scoped>
#tda-params {
}
</style>
