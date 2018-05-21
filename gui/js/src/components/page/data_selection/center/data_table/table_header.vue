<template>
  <div class="table_header">
    <tr>
      <th>histogram</th>
      <th class='flex_grow_2'>column</th>
      <th :class='{active: selected_column === "dtype"}'
        @click='select_column("dtype")'>data type</th>
      <th :class='{active: selected_column === "mean"}'
        @click='select_column("mean")'>mean</th>
      <th :class='{active: selected_column === "var"}'
        @click='select_column("var")'>var</th>
      <th :class='{active: selected_column === "std"}'
        @click='select_column("std")'>std</th>
      <th :class='{active: selected_column === "min"}'
        @click='select_column("min")'>min</th>
      <th :class='{active: selected_column === "25%"}'
        @click='select_column("25%")'>25%</th>
      <th :class='{active: selected_column === "50%"}'
        @click='select_column("50%")'>50%</th>
      <th :class='{active: selected_column === "75%"}'
        @click='select_column("75%")'>75%</th>
      <th :class='{active: selected_column === "max"}'
        @click='select_column("max")'>max</th>
    </tr>
  </div>
</template>

<script>
import {mapState} from 'vuex'

export default {
  name: 'TableHeader',
  computed: mapState(['row', 'selected_column']),
  methods: {
    select_column: function(name) {
      this.$store.commit('set_select_column_name', {'name': name});
    },
  }
}
</script>

<style lang='scss' scoped>
.table_header{
  $all-selector-row-height: 34px;
  $background-color: #f8f8f8;
  $background-color-active: #f0f0f0;
  $border-color: #cccccc;
  $table-font-size: 10px;

  width: 100%;
  border: none;

  tr {
    border: none;
    border-bottom: 1px solid $border-color;

    th {
      height: $all-selector-row-height;
      padding: 0;
      font-size: $table-font-size;
      font-weight: normal;
      background-color: $background-color;
      text-align: center;
      line-height: $all-selector-row-height;
      border: none;
    }
  }
  .flex_grow_2 {
    flex-grow: 2;
  }
  .active {
    background-color: $background-color-active;
  }
}
</style>
