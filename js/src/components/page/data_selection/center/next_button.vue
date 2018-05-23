<template>
  <div id='next_button'>
    <button
      :disabled="$store.state.data_header.length == 0"
      @click='go_next'>
      Next
    </button>
  </div>
</template>

<script>
export default {
  name: 'NextButton',
  methods: {
    go_next: function() {
      if(this.$store.state.file_data) {
        this.$router.push({ path: '/topology' });
      }else{
        alert("File is not loaded.");
        return;
      }
    },
  }
}
</script>

<style lang='scss' scoped>
#next_button {
}
</style>
