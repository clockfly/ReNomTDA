<template>
  <div id="file_loader">
    <div class="input-group vertical">
      <select v-model="file_id">
        <option v-for="(item, index) in files" :key="index" :value="item.id">{{ item.name }}</option>
      </select>
    </div>

    <button class="load_file_button" v-on:click="load_file">Load file</button>
  </div>
</template>

<script>
export default {
  name: "TopologyFileLoader",
  computed: {
    file_id: {
      get: function() {
        return this.$store.state.topology.file_id;
      },
      set: function(val) {
        this.$store.commit("set_file_id", {
          "file_id": val,
        });
      },
    },
    files() {
      return this.$store.state.topology.files;
    }
  },
  methods: {
    load_file: function() {
      this.$store.dispatch('load_file', {
        'file_id': this.file_id
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