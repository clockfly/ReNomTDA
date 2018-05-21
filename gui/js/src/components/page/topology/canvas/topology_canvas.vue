<template>
  <div id="topology-canvas">
    <div class="canvas-area">
      <div v-for="(topology, index) in $store.state.topologies"
        :class="'canvas'+index">
        <d3-canvas :id="index" v-if="topology.nodes.length > 0"></d3-canvas>
        <div class="histogram-area" v-if="$store.state.file_id != ''">
          <histogram :id="index"></histogram>
          <color-selector :id="index"></color-selector>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import D3Canvas from './d3_canvas.vue'
import Histogram from './histogram.vue'
import ColorSelector from './color_selector.vue'

export default {
  name: "TopologyCanvas",
  components: {
    "d3-canvas": D3Canvas,
    "histogram": Histogram,
    "color-selector": ColorSelector,
  },
}
</script>

<style lang="scss" scoped>
#topology-canvas {
  $canvas-header-height: 32px;

  width: 100%;
  height: 100%;

  .canvas-header-area {
    width: 100%;
    height: $canvas-header-height;
  }
  .canvas-area{
    display: flex;
    width: 100%;
    height: 100%;
  }
  .canvas0, .canvas1 {
    position: relative;
    width: 50%;
    height: 100%;
    margin: 0;
    padding: 0;
  }
  .canvas1 {
    border-left: 1px solid #cccccc;
  }
  .histogram-area {
    position: absolute;
    top: 0;
    right: 0;
    width: 150px;
    height: 100px;
  }
}
</style>
