<template>
  <div id="k-means">
    <div class="input-group vertical">
      <label for="k">K: {{ k }}</label>
      <input type="range" min="2" max="10" step="1" class="slider" v-model="k">
    </div>
  </div>
</template>

<script>
export default {
  name: "KMeans",
  props: {
    "columnIndex": {
      type: Number,
      require: true,
    }
  },
  computed: {
    "k": {
      get: function() {
        return this.$store.state.topologies[this.columnIndex].k;
      },
      set: function(val) {
        this.$store.commit("set_k", {
          "index": this.columnIndex,
          "val": val,
        })
      },
    }
  }
}
</script>

<style lang="scss" scoped>
#k-means {
}
</style>
