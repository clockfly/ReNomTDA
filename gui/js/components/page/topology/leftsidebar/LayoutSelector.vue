<template>
  <div id="layout_selector">
    <select v-model="layout" v-on:change="set_layout">
      <option v-for="(l, index) in layouts" :value="index">{{l}}</option>
    </select>
  </div>
</template>

<script>
export default {
  name: "TopologyAlgorithmSelector",
  data: function() {
    return {
      layouts: ["1 Column", "2 Columns"],
      layout: 0,
    }
  },
  methods: {
    set_layout: function() {
      this.$store.commit('set_layout', {
        'layout_columns': this.layout
      });
    }
  }
}
</script>
