<template>
  <div class="table_row">
    <tr>
      <td>
        <div class="histogram_area" v-if='isnum'>
          <histogram :id='index' :histdata='histdata'></histogram>
        </div>
      </td>
      <td class='flex_grow_2'>{{ colname }}</td>
      <td :class='{active: selected_column === "dtype"}'>
        <span v-if='isnum'>Number</span>
        <span v-if='!isnum'>Text</span>
      </td>
      <td :class='{active: selected_column === "mean"}'>
        <span v-if='isnum'>{{ mean }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "var"}'>
        <span v-if='isnum'>{{ vari }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "std"}'>
        <span v-if='isnum'>{{ std }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "min"}'>
        <span v-if='isnum'>{{ min }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "25%"}'>
        <span v-if='isnum'>{{ percentile25 }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "50%"}'>
        <span v-if='isnum'>{{ percentile50 }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "75%"}'>
        <span v-if='isnum'>{{ percentile75 }}</span>
        <span v-if='!isnum'>-</span>
      </td>
      <td :class='{active: selected_column === "max"}'>
        <span v-if='isnum'>{{ max }}</span>
        <span v-if='!isnum'>-</span>
      </td>
    </tr>
  </div>
</template>

<script>
import {mapState} from 'vuex'
import Histogram from './histogram.vue'

export default {
  name: 'TableRowWithoutTimeseries',
  components: {
    'histogram': Histogram,
  },
  props: ['index', 'isnum', 'histdata', 'colname', 'mean',
          'vari', 'std', 'min', 'percentile25',
          'percentile50', 'percentile75', 'max'],
          computed: mapState(['selected_column']),
}
</script>

<style lang='scss' scoped>
.table_row {
  $table-row-height: 91px;
  $background-color: #f8f8f8;
  $background-color-active: #f0f0f0;
  $border-width: 1px;
  $border-color: #cccccc;
  $table-font-size: 10px;

  width: 100%;
  height: calc(#{$table-row-height}+#{$border-width});
  border: none;
  border-bottom: $border-width solid $border-color;
  box-sizing: border-box;

  td {
    height: $table-row-height;
    padding: 0;
    font-size: $table-font-size;
    text-align: center;
    line-height: $table-row-height;
    border: none;

    select{
      padding-left: 4px;
      padding-right: 4px;
      font-size: $table-font-size;
    }

    .select_check_box {
      margin-left: 10px;
      padding-top: 18px;
      input[type='checkbox']+label{
        margin-top: 10px;
      }
      input[type='checkbox']+label:before {
        bottom: 0.55rem;
        left: 0.5rem;
        width: $table-font-size;
        height: $table-font-size;
      }
      input[type='checkbox']+label:after {
        left: 0.6rem;
        width: $table-font-size;
        height: $table-font-size;
        background-color: #666;
      }
    }

    .histogram_area {
      margin-top: 10px;
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
