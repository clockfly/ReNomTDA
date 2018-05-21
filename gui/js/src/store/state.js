export default {
  // files info dictionaly
  files: {},
  // selected file id
  file_id: '',
  // loading flag
  loading: [false, false],

  // highlighted column name
  selected_column: undefined,

  // file summary data
  row: 0,
  columns: 0,

  // file header
  data_header: [],

  // number columns index
  number_index: undefined,

  // histogram data
  hist_data: undefined,

  // statistic data
  data_mean: undefined,
  data_var: undefined,
  data_std: undefined,
  data_min: undefined,
  data_25percentile: undefined,
  data_50percentile: undefined,
  data_75percentile: undefined,
  data_max: undefined,

  /*
  topology pages
  */
  // selected target column index
  target_index: '',

  // file_data
  file_data: undefined,

  // column_data
  number_columns: undefined,
  text_columns: undefined,

  // topology hist
  topo_hist: undefined,

  // algorithms
  algorithms: ["PCA", "TSNE", "Isomap", "None"],
  modes: ["Scatter Plot", "Clustering(教師なし)", "Clustering(教師あり)", "TDA"],
  unsupervised_clusterings: ["K-Means", "DBSCAN"],
  supervised_clusterings: ["K-NearestNeighbor"],

  // topologies
  topologies: [{
    algorithm: 0,
    mode: 0,
    clustering_algorithm: 0,
    train_size: 0.9,
    k: 3,
    eps: 1,
    min_samples: 1,
    resolution: 10,
    overlap: 0.5,
    color_index: 0,
    hypercubes: [],
    point_cloud: [],
    nodes: [],
    node_sizes: [],
    edges: [],
    colors: [],
    train_index: [],
    test_index: [],
  }, {
    algorithm: 0,
    mode: 0,
    clustering_algorithm: 0,
    train_size: 0.9,
    k: 3,
    eps: 1,
    min_samples: 1,
    resolution: 10,
    overlap: 0.5,
    color_index: 0,
    hypercubes: [],
    point_cloud: [],
    nodes: [],
    node_sizes: [],
    edges: [],
    colors: [],
    train_index: [],
    test_index: [],
  }],

  // search conditions
  search_type: "and",
  search_conditions: [{
    "data_type": "number",
    "operator": "=",
    "column": 0,
    "value": 0,
  }],

  // click node
  click_node_data_ids: [],

  // show topology setting modal
  show_setting_modal: false,

  // show search condition modal
  show_search_modal: false,

  // visualize model
  visualize_mode: 0,
}
