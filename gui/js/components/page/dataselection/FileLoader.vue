<template>
  <div id="file_loader">
    <div class="input-group vertical">
      <input type="text" id="filename" placeholder="input file name" v-model="filename">
    </div>

    <button class="load_file_button" v-on:click="load_file">Load file</button>
  </div>
</template>

<script>
export default {
  name: "TopologyFileLoader",
  data: function() {
    return {
      filename: ""
    }
  },
  methods: {
    load_file: function() {
      this.$store.dispatch('load_file', {
        'filename': this.filename
      });
    }
  }
}
</script>

<style lang="scss">
#file_loader {
  .load_file_button {
    width: 100%;
    margin: 8px 0 0 0;
    padding: 8px;
  }
}
</style>