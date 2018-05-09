<template>
  <div class="color-selector">
    <div class="input-group vertical">
      <label for="color">colored by</label>
      <select v-model="color_index">
        <option v-for="(v,index) in $store.state.number_index" :value="index">{{$store.state.data_header[index]}}</option>
      </select>
    </div>
  </div>
</template>

<script>
export default {
  name: "ColorSelector",
  props: {
    "id": {
      type: Number,
      required: true
    }
  },
  computed: {
    color_index: {
      get: function() {
        return this.$store.state.topologies[this.id].color_index;
      },
      set: function(val) {
        this.$store.commit("set_color_index", {
          "index": this.id,
          "val": val,
        });
      }
    },
  }
}
</script>

<style>
</style>
