<template>
  <div id="canvas-header">
    <div class="setting-button-area">
      <button class="show-setting-button" @click="showSettingModal">
        <i class="fa fa-cog" aria-hidden="true"></i>
      </button>
    </div>

    <div class="search-button-area">
      <button class="show-search-button" @click="showSearchModal">
        search settings
      </button>
      <button class="search-reset-button" @click="resetSearch">
        clear search
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

  .search-button-area {
    margin-left: auto;
  }
}
</style>
