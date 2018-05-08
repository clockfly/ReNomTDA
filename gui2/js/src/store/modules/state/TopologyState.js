let TopologyState = {
  // ロード中はtrue
  loading: false,

  // ファイル名
  file_id: 0,

  // file list
  files: [],

  // カテゴリデータのラベル
  categorical_data_labels: undefined,

  // 数値データのラベル
  numerical_data_labels: undefined,
  numerical_data: undefined,
  numerical_data_mins: undefined,
  numerical_data_maxs: undefined,
  numerical_data_means: undefined,

  // 計算カラムと色のカラムのインデックス
  create_topology_index: [],
  colorize_topology_index: [],

  // 0: 1カラム, 1:2カラム
  layout_columns: 0,

  // アルゴリズムのパラメータ
  algorithm_index: 0,
  algorithm_index_col2: 0,

  // TDAのパラメータ
  mode: 0,
  resolution: 25,
  overlap: 0.5,
  eps: 0.5,
  min_samples: 3,
  color_index: 0,
  clustering_index: 0,
  class_count: 3,
  train_size: 0.8,
  neighbors: 5,
  train_index: undefined,

  // TDAのパラメータ、カラム２
  mode_col2: 0,
  resolution_col2: 25,
  overlap_col2: 0.5,
  eps_col2: 0.5,
  min_samples_col2: 3,
  color_index_col2: 0,
  clustering_index_col2: 0,
  class_count_col2: 3,
  train_size_col2: 0.8,
  neighbors_col2: 5,
  train_index_col2: undefined,

  pca_result: undefined,

  // 表示に関するパラメータ、真ん中上のヘッダーで変更する
  show_histogram: false,
  show_spring: false,

  // 検索文字列
  search_txt: "",

  // トポロジーの計算結果
  nodes: undefined,
  edges: undefined,
  colors: undefined,
  sizes: undefined,

  // ２列目の計算結果
  nodes_col2: undefined,
  edges_col2: undefined,
  colors_col2: undefined,
  sizes_col2: undefined,

  // ヒストグラム表示に使うデータ
  data_color_values: undefined,
  data_color_values_col2: undefined,
  statistic_value: undefined,
  statistic_value_col2: undefined,

  // ノードをクリックした時の値
  click_node_index: undefined,
  node_categorical_data: undefined,
  node_data: undefined,

  // 検索をカテゴリでするか、数値でするか
  show_category: 0,
}

export default TopologyState