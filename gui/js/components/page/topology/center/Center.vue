<template>
  <div id="center">
    <div id="canvas_header">
      <div class="canvas_header_button_area">
        <span>全画面・2画面:</span>
        <button id="split_layout_button" v-on:click="split_layout">
          <i class="fa fa-window-restore" aria-hidden="true"></i>
        </button>

        <span>ヒストグラム:</span>
        <button id="show_histogram_button" v-on:click="show_histogram">
          <i class="fa fa-bar-chart" aria-hidden="true"></i>
        </button>

        <span>TDA 3D:</span>
        <button id="show_spring_button" v-on:click="show_spring">
          <i class="fa fa-cube" aria-hidden="true"></i>
        </button>

        <span>リセット:</span>
        <button id="reset_button" v-on:click="reset">
          <i class="fa fa-times" aria-hidden="true"></i>
        </button>
      </div>

      <div class="pca_result" v-if="show_pca_result">
        <span v-if="pca_result" class="pca_result_span">
          第一主成分：{{ calc_labels[pca_result.top_index[0][2]] }}:{{ pca_result.axis[0][pca_result.top_index[0][2]] }}, {{ calc_labels[pca_result.top_index[0][1]] }}:{{ pca_result.axis[0][pca_result.top_index[0][1]] }}, {{ calc_labels[pca_result.top_index[0][0]] }}:{{ pca_result.axis[0][pca_result.top_index[0][0]] }}</span>
        <span v-if="pca_result" class="pca_result_span">
          第二主成分：{{ calc_labels[pca_result.top_index[1][2]] }}:{{ pca_result.axis[1][pca_result.top_index[1][2]] }}, {{ calc_labels[pca_result.top_index[1][1]] }}:{{ pca_result.axis[1][pca_result.top_index[1][1]] }}, {{ calc_labels[pca_result.top_index[1][0]] }}:{{ pca_result.axis[1][pca_result.top_index[1][0]] }}</span>
        <span v-if="pca_result" class="pca_result_span">寄与率：{{ pca_result.contribution_ratio }}</span>
      </div>
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
    },
    pca_result() {
      return this.$store.state.topology.pca_result
    },
    show_pca_result() {
      if(this.$store.getters.layout_columns == 0) {
        if(this.$store.state.topology.algorithm_index == 0){
          return true
        }
      }else if(this.$store.getters.layout_columns == 1) {
        if(this.$store.state.topology.algorithm_index == 0 || this.$store.state.topology.algorithm_index_col2 == 0){
          return true
        }
      }
      return false
    },
    calc_labels() {
      return this.$store.getters.calc_labels;
    },
  },
  methods: {
    split_layout: function() {
      document.getElementById("split_layout_button").classList.toggle('inverse');
      this.$store.commit('set_layout', {
        'layout_columns': (this.$store.state.topology.layout_columns + 1) % 2
      });
    },
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
    .canvas_header_button_area {
      float: left;
      height: 100%;
    }
    #split_layout_button, #show_histogram_button, #show_spring_button, #reset_button {
      height: 100%;
      margin-top: 4px;
      margin-left: 0px;
      padding: 0 8px;
    }
    .pca_result {
      .pca_result_span {
        display: block;
        line-height: 0.9rem;
        font-size: 0.8rem;
      }
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