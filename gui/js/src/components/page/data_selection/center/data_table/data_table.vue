<template>
  <div id='data_table'>
    <table class='scrollable'>
      <thead>
        <table-header></table-header>
      </thead>

      <tbody>
        <table-row-without-ts
          v-for='(l, index) in data_header'
          :key='index'
          :index='index'
          :isnum='number_index[index]'
          :histdata='hist_data[index]'
          :colname='l'
          :mean='round(data_mean[index])'
          :vari='round(data_var[index])'
          :std='round(data_std[index])'
          :min='round(data_min[index])'
          :percentile25='round(data_25percentile[index])'
          :percentile50='round(data_50percentile[index])'
          :percentile75='round(data_75percentile[index])'
          :max='round(data_max[index])'>
        </table-row-without-ts>
      </tbody>
    </table>
  </div>
</template>

<script>
import {mapState} from 'vuex'
import TableHeader from './table_header.vue'
import TableRowWithoutTimeseries from './table_row_without_timeseries.vue'

export default {
  name: 'DataTable',
  components: {
    'table-header': TableHeader,
    'table-row-without-ts': TableRowWithoutTimeseries,
  },
  computed: mapState([
      'row',
      'interpolate_items',
      'show_time_series',
      'number_index',
      'number_data',
      'hist_data',
      'data_header',
      'data_mean',
      'data_var',
      'data_std',
      'data_min',
      'data_25percentile',
      'data_50percentile',
      'data_75percentile',
      'data_max']),
  methods: {
    round: function(val) {
      return Math.round(val*100)/100;
    },
  }
}
</script>

<style lang='scss' scoped>
#data_table {
  $background-color: #f8f8f8;
  $border-color: #cccccc;
  $table-row-height: 92px;
  $all-selector-row-height: 34px;
  $timeseries-text-row-height: 40px;
  $table-font-size: 10px;

  width: 100%;
  height: 100%;
  padding: 0 48px;

  table.scrollable {
    padding: 0;
  }
  table {
    max-height: 100%;
    background-color: $background-color;
    thead, tbody {
      background-color: $background-color;
      border: none;
    }
  }
}
</style>
