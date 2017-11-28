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

<style lang="scss">
header {
  $header_height: 50px;
  $header_color_from: #223347;
  $header_color_to: #1c4254;

  height: $header_height;
  margin: 0;
  padding: 0;
  background: -moz-linear-gradient($header_color_from, $header_color_to);
  background: -webkit-gradient(linear, left top, left bottom, from($header_color_from), to($header_color_to));
  font-family:"Arial Black";

  .header_title {
    height: 100%;
    padding-left: 20px;
    color: #eeeeee;
    font-size: 18px;
    line-height: $header_height;
  }
}
</style>