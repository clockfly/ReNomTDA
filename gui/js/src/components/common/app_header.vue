<template>
  <header>
    <div class="header_title">ReNom TDA</div>
  </header>
</template>

<script>
export default {
  name: "AppHeader"
}
</script>

<style lang="scss" scoped>
header {
  $header-height: 44px;
  $header-color: #1e264b;
  $font-size: 16px;
  $font-color: #ffffff;

  $content-margin: 8px;

  display: flex;
  position: fixed;
  top: 0px;
  left: 0px;
  width: 100vw;
  height: $header-height;

  margin: 0;
  padding: 0;

  background-color: $header-color;
  z-index: 999;

  .header_title {
    margin-left: 16px;
    font-size: $font-size;
    font-weight: 500;
    line-height: $header-height;
    color: $font-color;
  }
}
</style>
