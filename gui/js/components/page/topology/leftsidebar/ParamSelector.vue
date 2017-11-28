<template>
  <div id="param_selector">
    <div class="input-group vertical">
      <label for="mode">Analysis Mode:</label>
      <select v-model="mode" v-on:change="set_mode">
        <option v-for="(item, index) in modes" :value="index">{{item}}</option>
      </select>
    </div>

    <div class="input-group vertical" v-if="mode == 1">
      <label for="clustering_index">Clustering Algorithm:</label>
      <select v-model="clustering_index" v-on:change="set_clustering_index">
        <option v-for="(item, index) in clustering_algorithms" :value="index">{{item}}</option>
      </select>
    </div>

    <div class="mapping_params" v-if="mode == 2">
      <p>Mapping Parameters</p>
      <div class="input-group vertical">
        <label for="resolution">resolution: {{resolution}}</label>
        <input type="range" min="5" max="50" step="5" class="slider" v-model="resolution" v-on:change="set_resolution">
      </div>

      <div class="input-group vertical">
        <label for="overlap">overlap: {{overlap}}</label>
        <input type="range" min="0.1" max="1" step="0.1" class="slider" v-model="overlap" v-on:change="set_overlap">
      </div>
    </div>

    <div class="input-group vertical" v-if="mode == 1 && clustering_index == 0">
      <p>Clustering Parameters</p>
      <div class="input-group vertical">
        <label for="epsilon">K: {{ class_count }}</label>
        <input type="range" min="2" max="10" step="1" class="slider" v-model="class_count" v-on:change="set_class_count">
      </div>
    </div>

    <div class="dbscan_params" v-if="(mode == 1 && clustering_index == 1) || mode == 2">
      <p>Clustering Parameters</p>
      <div class="input-group vertical">
        <label for="epsilon">epsilon: {{eps}}</label>
        <input type="range" min="0.01" max="1" step="0.01" class="slider" v-model="eps" v-on:change="set_epsilon">
      </div>

      <div class="input-group vertical">
        <label for="min_samples">min samples: {{min_samples}}</label>
        <input type="range" min="1" max="5" step="1" class="slider" v-model="min_samples" v-on:change="set_min_samples">
      </div>
    </div>

    <div class="input-group vertical" v-if="mode == 1 && clustering_index > 1">
      <p>Clustering Parameters</p>
      <div class="input-group vertical">
        <label for="epsilon">train data size: {{ train_size }}</label>
        <input type="range" min="0.1" max="0.9" step="0.1" class="slider" v-model="train_size" v-on:change="set_train_size">
      </div>
    </div>

    <div class="input-group vertical" v-if="mode == 1 && clustering_index == 2">
      <div class="input-group vertical">
        <label for="epsilon">K: {{ neighbors }}</label>
        <input type="range" min="3" max="11" step="2" class="slider" v-model="neighbors" v-on:change="set_neighbors">
      </div>
    </div>

    <div class="input-group vertical" v-if="mode == 0 || (mode == 1 && clustering_index > 1) || mode == 2">
      <label for="color_index">colored by:</label>
      <select v-model="color_index" v-on:change="set_color_index">
        <option v-for="(item, index) in color_labels" :value="index">{{item}}</option>
      </select>
    </div>
  </div>
</template>

<script>
export default {
  name: "TopologyParamSelector",
  props: ["column"],
  data: function() {
    return {
      modes: ["Scatter Plot", "Clustering", "TDA"],
      clustering_algorithms: ["K-means", "DBSCAN", "K-NearestNeighbor", "SVM", "Random Forest"],
      mode: 0,
      resolution: 25,
      overlap: 0.5,
      eps: 0.5,
      min_samples: 3,
      color_index: 0,
      clustering_index: 0,
      class_count: 3,
      train_size: 0.8,
      neighbors: 5
    }
  },
  computed: {
    color_labels(){
      return this.$store.getters.color_labels
    }
  },
  methods: {
    set_mode: function() {
      this.$store.commit('set_mode', {
        'mode': this.mode,
        'column': this.column
      });
    },
    set_resolution: function() {
      this.$store.commit('set_resolution', {
        'resolution': this.resolution,
        'column': this.column
      });
    },
    set_overlap: function() {
      this.$store.commit('set_overlap', {
        'overlap': this.overlap,
        'column': this.column
      });
    },
    set_epsilon: function() {
      this.$store.commit('set_epsilon', {
        'eps': this.eps,
        'column': this.column
      });
    },
    set_min_samples: function() {
      this.$store.commit('set_min_samples', {
        'min_samples': this.min_samples,
        'column': this.column
      });
    },
    set_color_index: function() {
      this.$store.commit('set_color_index', {
        'color_index': this.color_index,
        'column': this.column
      });
    },
    set_clustering_index: function() {
      this.$store.commit('set_clustering_index', {
        'clustering_index': this.clustering_index,
        'column': this.column
      });
    },
    set_class_count: function() {
      this.$store.commit('set_class_count', {
        'class_count': this.class_count,
        'column': this.column
      });
    },
    set_train_size: function() {
      this.$store.commit('set_train_size', {
        'train_size': this.train_size,
        'column': this.column
      });
    },
    set_neighbors: function() {
      this.$store.commit('set_neighbors', {
        'neighbors': this.neighbors,
        'column': this.column
      });
    }
  }
}
</script>
