<template>
  <div id="nodedata">
    <p>{{ categoricaldata[show_category] }}</p>
    <p class="data" v-if="!show_more">{{ labels[colorize_topology_index[color_index]] }}: {{ data[colorize_topology_index[color_index]] }}</p>

    <a class="show_more" v-if="!show_more" v-on:click="change_show_more">詳細を開く</a>
    <p class="data" v-if="show_more" v-for="(d, i) in data">{{ labels[i] }}: {{ d }}</p>
    <a class="show_more" v-if="show_more" v-on:click="change_show_more">閉じる</a>
  </div>
</template>

<script>
export default {
  name: "TopologyNodeData",
  props: ["data", "index", "categoricaldata"],
  data: function() {
    return {
      show_more: false
    }
  },
  computed: {
    show_category() {
      return this.$store.state.topology.show_category
    },
    color_index() {
      return this.$store.state.topology.color_index
    },
    labels() {
      return this.$store.state.topology.numerical_data_labels
    },
    colorize_topology_index() {
      return this.$store.state.topology.colorize_topology_index
    }
  },
  methods: {
    change_show_more: function() {
      this.show_more = !this.show_more;
    }
  }
}
</script>

<style lang="scss">
#nodedata {
  margin: 10px 10%;
  border-bottom: 1px solid #ccc;
  .show_more {
    color: #999;
  }
  .data {
    font-size: 14px;
  }
}
</style>