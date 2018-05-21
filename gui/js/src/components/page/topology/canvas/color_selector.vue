<template>
  <div class="color-selector">
    <div class="input-group vertical">
      <label for="color">colored by</label>
      <select class="padding-left-8" v-model="color_index">
        <option v-for="(v,index) in number_columns" :value="index">{{v}}</option>
      </select>
    </div>
  </div>
</template>

<script>
import {mapState} from 'vuex'

export default {
  name: "ColorSelector",
  props: {
    "id": {
      type: Number,
      required: true
    }
  },
  computed: {
    ...mapState(['number_index', 'number_columns']),
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
