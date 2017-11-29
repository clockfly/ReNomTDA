<template>
  <div id="center">
    <div class="number_data_area">
      <h3>Number Data</h3>
      <table>
        <thead>
          <tr>
            <th>カラム名</th>
            <th>最小値</th>
            <th>最大値</th>
            <th>平均値</th>
            <th>ヒストグラム</th>
            <th>計算に使う</th>
            <th>色に使う</th>
          </tr>
        </thead>
        <tbody>
            <tr>
              <td>全選択</td>
              <td></td>
              <td></td>
              <td></td>
              <td></td>
              <td>
                <div class="input-group">
                  <input type="checkbox" id="calc_all" :value="9999" v-model="create_all">
                  <label for="calc_all"></label>
                </div>
              </td>
              <td>
                <div class="input-group">
                  <input type="checkbox" id="color_all" :value="9999" v-model="colorize_all">
                  <label for="color_all"></label>
                </div>
              </td>
            </tr>
            <tr v-for="(label, index) in numerical_data_labels" :key="index">
              <td data-label="column_name">{{ label }}</td>
              <td data-label="data_min">{{ data_mins[index] }}</td>
              <td data-label="data_min">{{ data_maxs[index] }}</td>
              <td data-label="data_min">{{ data_means[index] }}</td>
              <td></td>
              <td data-label="check_calc">
                <div class="input-group">
                  <input type="checkbox" :id="'calc'+index" :value="index" v-model="create_topology_index">
                  <label :for="'calc'+index"></label>
                </div>
              </td>
              <td data-label="check_color">
                <div class="input-group">
                  <input type="checkbox" :id="'color'+index" :value="index" v-model="colorize_topology_index">
                  <label :for="'color'+index"></label>
                </div>
              </td>
            </tr>
        </tbody>
      </table>
    </div>

    <div class="text_data_area">
      <h3>Text Data</h3>
      <p v-for="(label, index) in categorical_data_labels" :key="index">{{ label }}</p>
    </div>

    <div class="next_button_area">
      <button class="next_button" v-on:click="go_topology">Next</button>
    </div>
  </div>
</template>

<script>
export default {
  name: "DataSelectionCenter",
  data: function() {
    return {
      create_topology_index: [],
      colorize_topology_index: [],
      create_all: false,
      colorize_all: false
    }
  },
  computed: {
    categorical_data_labels() {
      return this.$store.state.topology.categorical_data_labels;
    },
    numerical_data_labels() {
      return this.$store.state.topology.numerical_data_labels;
    },
    data_mins() {
      return this.$store.state.topology.numerical_data_mins;
    },
    data_maxs() {
      return this.$store.state.topology.numerical_data_maxs;
    },
    data_means() {
      return this.$store.state.topology.numerical_data_means;
    }
  },
  watch: {
    create_all: function(newVal, oldVal) {
      this.check_calc_all(newVal);
    },
    colorize_all: function(newVal, oldVal) {
      this.check_color_all(newVal);
    }
  },

  created: function() {
    this.create_topology_index = this.$store.state.topology.create_topology_index;
    this.colorize_topology_index = this.$store.state.topology.colorize_topology_index;
  },
  methods: {
    check_calc_all: function(check) {
      this.create_topology_index = [];
      if (check) {
        for (let i = 0; i < this.numerical_data_labels.length; i++){
          this.create_topology_index.push(i);
        }
      }
    },
    check_color_all: function(check) {
      this.colorize_topology_index = [];
      if (check) {
        for (let i = 0; i < this.numerical_data_labels.length; i++){
          this.colorize_topology_index.push(i);
        }
      }
    },
    go_topology: function() {
      if(this.create_topology_index.length > 1 && this.colorize_topology_index.length > 0) {
        this.$store.commit('set_selected_index', {
          'create_topology_index': this.create_topology_index,
          'colorize_topology_index': this.colorize_topology_index
        });
        this.$router.push({ path: '/topology' });
      }else{
        alert("Please select calculate columns & colorize columns.")
        return
      }
    }
  }
}
</script>

<style lang="scss">
#center {
  height: 100%;
  margin: 0;
  padding: 0;

  .number_data_area, .text_data_area, .next_button_area {
    width: 80%;
    margin: 20px auto;
  }
  .next_button_area {
    position: relative;
  }
  .next_button {
    position: absolute;
    right: 0;
  }
}
</style>