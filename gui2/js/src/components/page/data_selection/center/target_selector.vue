<template>
  <div id='target_selector'>
    <div class='input-group fluid'>
      <label for='select_target'>Select Target</label>
      <select id='select_target' class='select_target' v-model='target_index'>
        <option value=''>None</option>
        <option v-for='(item, index) in data_header' :key='index' :value='index'>{{ item }}</option>
      </select>
    </div>
  </div>
</template>

<script>
import {mapState} from 'vuex'

export default {
  name: 'TargetSelector',
  computed: {
    ...mapState(['data_header']),
    target_index: {
      get: function() {
        return this.$store.state.target_index;
      },
      set: function(val) {
        this.$store.commit('set_target_index', {
          'target_index': val,
        });
      },
    },
  },
}
</script>

<style lang='scss' scoped>
#target_selector {
  display: flex;
  display: -webkit-flex;

  .select_target {
    padding-left: 8px;
    line-height: 24px;
  }
}
</style>
