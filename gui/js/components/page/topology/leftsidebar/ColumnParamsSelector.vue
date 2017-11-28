<template>
  <div id="column_params_selector">
    <keep-alive>
      <dimension-reduction-selector :column="column"></dimension-reduction-selector>
    </keep-alive>

    <keep-alive>
      <param-selector :column="column"></param-selector>
    </keep-alive>
  </div>
</template>

<script>
import DimensionReductionSelector from './DimensionReductionSelector.vue'
import ParamSelector from './ParamSelector.vue'

export default {
  name: "ColumnParamsSelector",
  props: ["column"],
  components: {
    'dimension-reduction-selector': DimensionReductionSelector,
    'param-selector': ParamSelector
  }
}
</script>
