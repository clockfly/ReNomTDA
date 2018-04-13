<template>
  <div id="app" class="container">
    <appheader></appheader>
    <router-view></router-view>
    <loadingmodal v-if="loading"></loadingmodal>
  </div>
</template>

<script>
import AppHeader from './common/AppHeader.vue'
import LoadingModal from './common/LoadingModal.vue'

export default {
  name: "app",
  components:{
    'appheader': AppHeader,
    'loadingmodal': LoadingModal
  },
  computed: {
    loading(){
        return this.$store.getters.loading_state;
    }
  },
}
</script>

<style lang="scss">
@import url('https://rsms.me/inter/inter-ui.css');
* {
  font-family: 'Inter UI', sans-serif;
}

.container {
  margin: 0;
  padding: 0;
}
</style>