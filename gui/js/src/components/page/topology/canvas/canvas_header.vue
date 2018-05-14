<template>
  <div id="canvas-header">
    <div class="setting-button-area margin-top-8 margin-left-16">
      <button class="show-setting-button" @click="showSettingModal">
        <i class="fa fa-cog" aria-hidden="true"></i>
      </button>
    </div>

    <div class="input-group margin-top-8 margin-left-16">
      <label for="visualize">visualize mode</label>
      <select class="padding-left-8" v-model="visualize_mode">
        <option v-for="(v,index) in visualize_modes" :value="index">{{v}}</option>
      </select>
    </div>

    <div class="search-button-area margin-top-4 flex-right">
      <button class="show-search-button"
        :disabled="$store.state.topologies[0].point_cloud.length == 0"
        @click="showSearchModal">
        search settings
      </button>

      <button class="search-reset-button margin-top-4 margin-right-16"
        :disabled="$store.state.topologies[0].point_cloud.length == 0"
        @click="resetSearch">
        clear search
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: "CanvasHeader",
  data: function() {
    return {
      "visualize_modes": ["normal", "spring", "spectral"],
    }
  },
  computed: {
    visualize_mode: {
      get: function() {
        return this.$store.state.visualize_mode;
      },
      set: function(val) {
        this.$store.commit("set_visualize_mode", {
          "val": val,
        });
      }
    }
  },
  methods: {
    showSettingModal: function() {
      this.$store.commit('set_setting_modal', {"is_show": true});
    },
    showSearchModal: function() {
      this.$store.commit('set_search_modal', {"is_show": true});
    },
    resetSearch: function() {
      this.$store.dispatch("search", {
        "index": 0,
        "search_type": "and",
        "conditions": []
      });
      this.$store.dispatch("search", {
        "index": 1,
        "search_type": "and",
        "conditions": []
      });
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
}
</style>
