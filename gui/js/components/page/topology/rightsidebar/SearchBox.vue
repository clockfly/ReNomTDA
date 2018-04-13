<template>
  <div id="search_box">
    <div class="input-group vertical">
      <button class="change_search_data_type" v-on:click="change_search_data_type">Change Search Data Type</button>

      <div v-if="!is_categorical">Type: Number</div>
      <div v-if="is_categorical">Type: Text</div>

      <div v-if="is_categorical">
        <select v-model="search_column_index">
          <option v-for="(item, index) in categorical_labels" :value="index">{{item}}</option>
        </select>

        <select v-model="operator_index">
          <option v-for="(item, index) in categorical_operators" :value="index">{{item}}</option>
        </select>
      </div>

      <div v-if="!is_categorical">
        <select v-model="search_column_index">
          <option v-for="(item, index) in numerical_labels" :value="index">{{item}}</option>
        </select>

        <select v-model="operator_index">
          <option v-for="(item, index) in numerical_operators" :value="index">{{item}}</option>
        </select>
      </div>

      <input id="search_txt_box" placeholder="search value" v-model="search_value" @keyup.delete="deletetext">
    </div>

    <button class="search" v-on:click="search">Search</button>
  </div>
</template>

<script>
export default {
  name: "TopologySearchBox",
  data: function() {
    return {
      is_categorical: false,
      search_column_index: 0,
      operator_index: 0,
      categorical_operators: ["=", "like"],
      numerical_operators: ["=", ">", "<"],
      search_value: ""
    }
  },
  computed: {
    categorical_labels() {
      return this.$store.state.topology.categorical_data_labels;
    },
    numerical_labels() {
      return this.$store.getters.color_labels;
    }
  },
  methods: {
    search: function() {
      if(!this.$store.state.topology.nodes) {
        alert("Data is not visualized.");
        return;
      }

      // check input value
      if(!this.is_categorical && !isFinite(this.search_value)){
        alert("If you search number data, you can only input number data");
        return;
      }

      // 検索文字をbase64に変換
      let text_encoder = new TextEncoder();
      const search_txt = text_encoder.encode(this.search_value);

      this.$store.dispatch('search_topology', {
        'is_categorical': this.is_categorical,
        'search_column_index': this.search_column_index,
        'operator_index': this.operator_index,
        'search_txt': search_txt
      });
    },
    deletetext: function() {
      if (this.search_value == "") {
        this.search();
      }
    },
    change_search_data_type: function() {
      this.is_categorical = !this.is_categorical;
      this.search_column_index = 0;
      this.operator_index = 0;
      this.search_value = "";
    }
  }
}
</script>

<style lang="scss">
</style>