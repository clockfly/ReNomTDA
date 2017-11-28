<template>
  <div id="left_sidebar">
    <button class="go_prev" v-on:click="go_prev">データ選択に戻る</button>

    <div class="tabs stacked col-md-11 col-lg-11 tab_area">
      <input type="checkbox" id="layout_selector" aria-hidden="true">
      <label for="layout_selector" aria-hidden="true">Layout Selector</label>
      <layout-selector></layout-selector>

      <input type="checkbox" id="column_param_selector0" aria-hidden="true">
      <label for="column_param_selector0" aria-hidden="true">Column 0</label>
      <column-params-selector :column="'0'"></column-params-selector>

      <input v-if="layout_columns == 1" type="checkbox" id="column_param_selector1" aria-hidden="true">
      <label v-if="layout_columns == 1" for="column_param_selector1" aria-hidden="true">Column 1</label>
      <column-params-selector v-if="layout_columns == 1" :column="'1'"></column-params-selector>
    </div>

    <div class="col-md-11 col-lg-11 run_button_area">
      <button class="run_button" v-on:click="run">
        <span class="run_icon"><i class="fa fa-play" aria-hidden="true"></i></span>RUN
      </button>
    </div>
  </div>
</template>

<script>
import LayoutSelector from './LayoutSelector.vue'
import ColumnParamsSelector from './ColumnParamsSelector.vue'

export default {
  name: "TopologyLeftSideBar",
  components: {
    'layout-selector': LayoutSelector,
    'column-params-selector': ColumnParamsSelector
  },
  computed: {
    layout_columns() {
      return this.$store.getters.layout_columns;
    }
  },
  methods: {
    go_prev: function() {
      this.$store.dispatch('reset_all');
      this.$router.push({ path: '/' });
    },
    run: function() {
      this.$store.dispatch('create_topology');
    }
  }
}
</script>

<style lang="scss">
#left_sidebar {
  margin: 0;
  padding: 0;
  background-color: #f5f5f5;
  .tab_area, .run_button_area {
    margin: 10px auto;
  }
  .run_button {
    width: 100%;
    margin: 0;
  }
  .run_icon {
    margin-right: 4px;
  }
}
</style>