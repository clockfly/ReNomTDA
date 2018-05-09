<template>
  <div id="app">
    <appheader></appheader>
    <router-view></router-view>
    <loadingmodal v-if="$store.state.loading"></loadingmodal>
  </div>
</template>

<script>
import AppHeader from '@/components/common/app_header.vue'
import LoadingModal from './common/loading_modal.vue'

export default {
  name: "app",
  components:{
    'appheader': AppHeader,
    'loadingmodal': LoadingModal
  }
}
</script>

<style lang="scss">
@import url('https://rsms.me/inter/inter-ui.css');

$input-line-height: 24px;
$input-font-size: 12px;
$font-color: #666;

* {
  $background-color: #f8f8f8;

  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  -o-box-sizing: border-box;
  -ms-box-sizing: border-box;
  box-sizing: border-box;

  font-family: 'Inter UI', sans-serif;
  color: $font-color;
  background-color: $background-color;
}
select {
  height: $input-line-height;
  margin: 0;
  padding: 0;
  font-size: $input-font-size;
  color: $font-color;
  border: 1px solid #ccc;
  box-sizing: border-box;
}
button {
  height: $input-line-height;
  margin: 0;
  padding: 0 $input-line-height;
  font-size: $input-font-size;
  line-height: $input-line-height;
  color: $font-color;
  border: 1px solid #ccc;
  background-color: #f3f3f1;
  box-sizing: border-box;
}
input[type='number'] {
  -moz-appearance: textfield;
  width: 64px;
  height: $input-line-height;
  margin: 0;
  padding: 0;
  padding-left: 8px;
  font-size: $input-font-size;
  line-height: $input-line-height;
  border: 1px solid #ccc;
  color: $font-color;
}
input[type='number']::-webkit-outer-spin-button,
input[type='number']::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
.flex_right {
  margin-left: auto;
}
@for $i from 1 through 15 {
  .margin_left_#{$i*4} {
    margin-left: 4px * $i;
  }
  .margin_right_#{$i*4} {
    margin-right: 4px * $i;
  }
  .margin_top_#{$i*4} {
    margin-top: 4px * $i;
  }
  .margin_bottom_#{$i*4} {
    margin-bottom: 4px * $i;
  }
}
</style>
