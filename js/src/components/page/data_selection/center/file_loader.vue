<template>
  <div id='file_loader'>
    <div class='input-group'>
      <select class='select_file' v-model='file_id'>
        <option disabled value=''>Select File</option>
        <option v-for='(item, index) in files' :key='index' :value='item.id'>{{ item.name }}</option>
      </select>
    </div>

    <button class='load_button' @click='load'>
      <i class="fa fa-file-o" aria-hidden='true'></i> Load File
    </button>
  </div>
</template>

<script>
import {mapState} from 'vuex'

export default {
  name: 'FileLoader',
  computed: {
    ...mapState(['files']),
    file_id: {
      get: function() {
        return this.$store.state.file_id;
      },
      set: function(val) {
        this.$store.commit('set_file_id', {
          'file_id': val,
        });
      },
    },
  },
  methods: {
    load: function() {
      this.$store.dispatch('load_file', {
        'file_id': this.file_id
      });
    }
  }
}
</script>

<style lang='scss' scoped>
#file_loader {
  display: -webkit-flex;
  display: flex;

  .select_file {
    padding-left: 16px;
    line-height: 24px;
  }
  .load_button {
    margin-left: 12px;
  }
}
</style>
