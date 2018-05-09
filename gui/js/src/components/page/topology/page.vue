<template>
  <div id="page" class="row">
    <div class="canvas-header-area">
      <canvas-header></canvas-header>
    </div>

    <div class="canvas-area">
      <topology-canvas></topology-canvas>
    </div>

    <div class="selected-node-detail-area">
      <select-table></select-table>
    </div>

    <setting-modal v-if="$store.state.show_setting_modal"></setting-modal>
  </div>
</template>

<script>
import CanvasHeader from './canvas/canvas_header.vue'
import SelectTable from './select_data_table/select_table.vue'
import SettingModal from './setting_modal/setting_modal.vue'
import TopologyCanvas from './canvas/topology_canvas.vue'

export default {
  name: "TopologyPage",
  components: {
    'canvas-header': CanvasHeader,
    'select-table': SelectTable,
    'setting-modal': SettingModal,
    'topology-canvas': TopologyCanvas
  },
  created: function() {
    if (this.$store.state.file_id === '') {
      this.$router.push({ path: '/' });
    }
  }
}
</script>

<style lang="scss" scoped>
#page {
  $header-height: 44px;
  $canvas-header-height: 44px;
  $select-node-height: 200px;

  height: calc(100vh - #{$header_height});
  margin-top: $header-height;

  .canvas-header-area {
    width: 100vw;
    height: $canvas-header-height;
  }

  .canvas-area {
    width: 100vw;
    height: calc(100% - #{$canvas-header-height} - #{$select-node-height});
  }

  .selected-node-detail-area {
    width: 100vw;
    height: $select-node-height;
  }
}
</style>
