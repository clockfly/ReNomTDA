<template>
  <div id='next_button'>
    <button @click='go_next'>
      Next
    </button>
  </div>
</template>

<script>
export default {
  name: 'NextButton',
  methods: {
    go_next: function() {
      this.$router.push({ path: '/topology' });
    },
  }
}
</script>

<style lang='scss' scoped>
#next_button {
}
</style>
