<template>
  <div id="center">
    <div id="canvas_header">
      <button id="show_histogram_button" v-on:click="show_histogram"><i class="fa fa-bar-chart" aria-hidden="true"></i></button>
      <button id="show_spring_button" v-on:click="show_spring"><i class="fa fa-expand" aria-hidden="true"></i></button>
      <button id="reset_button" v-on:click="reset"><i class="fa fa-refresh" aria-hidden="true"></i></button>
    </div>

    <!-- 1 columns -->
    <div v-if="columns == 0" class="canvas_area">
      <d3canvas v-if="nodes && colors" :id="canvas_id" :width="width" :height="height" :nodes="nodes" :edges="edges" :colors="colors" :sizes="sizes" :trainindex="train_index" :histid="hist_id" :columns="'0'" :resolution="resolution"></d3canvas>
    </div>

    <!-- 2 columns -->
    <div v-if="columns == 1" class="canvas_area">
      <div class="canvas_2column border-right">
        <d3canvas v-if="nodes && colors" :id="canvas2_id" :width="width*0.5" :height="height" :nodes="nodes" :edges="edges" :colors="colors" :sizes="sizes" :trainindex="train_index" :histid="hist_id" :columns="'0'" :resolution="resolution"></d3canvas>
      </div>
      <div class="canvas_2column">
        <d3canvas v-if="nodes_col2 && colors_col2" :id="canvas_id" :width="width*0.5" :height="height" :nodes="nodes_col2" :edges="edges_col2" :colors="colors_col2" :sizes="sizes_col2" :trainindex="train_index_col2" :histid="hist2_id" :columns="'1'" :resolution="resolution_col2"></d3canvas>
      </div>
    </div>
  </div>
</template>

<script>
import D3Canvas from './D3Canvas.vue'

export default {
  name: "TopologyCenter",
  data: function() {
    return {
      canvas_id: "canvas",
      canvas2_id: "canvas2",
      hist_id: "hist",
      hist2_id: "hist2"
    }
  },
  components: {
    'd3canvas': D3Canvas
  },
  computed: {
    width() {
      let e = document.getElementById("center");
      return e.clientWidth;
    },
    height() {
      let e = document.getElementById("center");
      return e.clientHeight*0.95;
    },
    columns() {
      return this.$store.getters.layout_columns
    },
    resolution() {
      return this.$store.state.topology.resolution
    },
    resolution_col2() {
      return this.$store.state.topology.resolution_col2
    },
    nodes(){
      return this.$store.state.topology.nodes
    },
    edges() {
      return this.$store.state.topology.edges
    },
    colors() {
      return this.$store.state.topology.colors
    },
    sizes() {
      return this.$store.state.topology.sizes
    },
    train_index() {
      return this.$store.state.topology.train_index
    },
    nodes_col2() {
      return this.$store.state.topology.nodes_col2
    },
    edges_col2() {
      return this.$store.state.topology.edges_col2
    },
    colors_col2() {
      return this.$store.state.topology.colors_col2
    },
    sizes_col2() {
      return this.$store.state.topology.sizes_col2
    },
    train_index_col2() {
      return this.$store.state.topology.train_index_col2
    }
  },
  methods: {
    show_spring: function() {
      document.getElementById("show_spring_button").classList.toggle('inverse');
      this.$store.commit('set_show_spring');
    },
    show_histogram: function() {
      document.getElementById("show_histogram_button").classList.toggle('inverse');
      this.$store.commit('set_show_histogram');
    },
    reset: function() {
      this.$store.dispatch('reset');
    }
  }
}
</script>

<style lang="scss">
#center {
  height: 100%;
  margin: 0;
  padding: 0;

  #canvas_header {
    width: 100%;
    height: 5%;
    #show_histogram_button, #show_spring_button, #reset_button {
      height: 90%;
      margin-top: 4px;
      padding: 0 8px;
    }
  }
  .canvas_area {
    height: 95%;
    width: 100%;
    svg {
      height: 100%;
      width: 100%;
    }
  }

  .canvas_2column {
    float: left;
    width: 50%;
    height: 100%;
    margin: 0;
    padding: 0;
    box-sizing:border-box;
  }

  .border-right {
    border-right: 2px solid #eeeeee;
  }
}
</style>