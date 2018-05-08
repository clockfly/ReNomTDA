export default {
  // files info dictionaly
  files: {},
  // selected file id
  file_id: '',

  // selected target column index
  target_index: '',

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

  // loading flag
  loading: false,

  // show topology setting modal
  show_setting_modal: false,
}
