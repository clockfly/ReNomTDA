<template>
  <div id="canvas-header">
    <div class="setting-button-area">
      <button class="show-setting-button" @click="showSettingModal">
        <i class="fa fa-cog" aria-hidden="true"></i>
      </button>
    </div>

    <div class="search-box-area">
      <search-box></search-box>
    </div>
  </div>
</template>

<script>
import SearchBox from './search_box.vue'

export default {
  name: "CanvasHeader",
  components: {
    "search-box": SearchBox,
  },
  methods: {
    showSettingModal: function() {
      this.$store.commit('set_setting_modal', {"is_show": true});
    }
  }
}
</script>

<style lang="scss" scoped>
#canvas-header {
  $border-color: #cccccc;

  display: flex;
  width: 100%;
  height: 100%;
  border-bottom: 1px solid $border-color;

  .search-box-area {
    width: 30%;
    margin-left: auto;
  }
}
</style>
