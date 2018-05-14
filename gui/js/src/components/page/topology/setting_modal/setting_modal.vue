<template>
  <div id="setting-modal">
    <div class="modal-background" @click="hideModal"></div>
    <div class="modal-content">
      <div class="modal-title">
        Topology Settings
      </div>

      <div class="modal-param-area margin-top-16">
        <common-params v-for="(t, index) in $store.state.topologies"
          :key="index" :columnIndex="index" class="column-pararm-area"></common-params>
      </div>

      <div class="modal-button-area margin-top-32">
        <button class="run-button" @click="run">
          <i class="fa fa-play" aria-hidden="true"></i> RUN
        </button>

        <button class="cancel-button" @click="hideModal">
          CANCEL
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import CommonParams from './common_params.vue'

export default {
  name: "SettingModal",
  components: {
    "common-params": CommonParams,
  },
  methods: {
    hideModal: function() {
      this.$store.commit("set_setting_modal", {"is_show": false});
    },
    run: function() {
      this.$store.dispatch("create", {"index": 0});
      this.$store.dispatch("create", {"index": 1});
    },
  }
}
</script>

<style lang="scss" scoped>
#setting-modal {
  $header_height: 35px;

  $modal-color: #000000;
  $modal-opacity: 0.5;

  $modal-content-width: 60%;
  $modal-content-height: 80%;
  $modal-content-bg-color: #fefefe;
  $modal-content-padding: 32px;

  $modal-title-font-size: 32px;
  $modal-sub-title-font-size: 24px;

  position: fixed;
  width: 100vw;
  height: calc(100vh - #{$header_height});

  .modal-background {
    width: 100%;
    height: 100%;
    background-color: $modal-color;
    opacity: $modal-opacity;
  }

  .modal-content {
    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);

    width: $modal-content-width;
    height: $modal-content-height;
    padding: $modal-content-padding;
    background-color: $modal-content-bg-color;
    opacity: 1;

    .modal-title {
      font-size: $modal-title-font-size;
    }

    .modal-param-area {
      display: flex;
      width: 100%;

      .column-pararm-area {
        width: 50%;
      }
    }
  }
}
</style>
