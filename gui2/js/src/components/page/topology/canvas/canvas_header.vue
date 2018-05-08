<template>
  <div id="canvas-header">
    <div class="setting-button-area">
      <button class="show-setting-button" @click="showSettingModal">
        <i class="fa fa-cog" aria-hidden="true"></i>
      </button>
    </div>
  </div>
</template>

<script>

export default {
  name: "CanvasHeader",
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
  border-bottom: 2px solid $border-color;
}
</style>
