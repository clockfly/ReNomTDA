<template>
  <div class="histogram">
    <svg :id="'hist'+id" width="150" height="60"></svg>
  </div>
</template>

<script>
export default {
  name: "Histgram",
  props: {
    "id": {
      type: Number,
      required: true
    }
  },
  computed: {
    color_index() {
      return this.$store.state.topologies[this.id].color_index;
    },
    file_data() {
      return this.$store.state.topo_hist;
    },
  },
  watch: {
    color_index: function() {
      this.draw_graph();
    }
  },
  mounted: function(){
    this.draw_graph();
  },
  methods: {
    interpolation_hsv: function(value, color_min, color_max) {
      let h;
      if(color_min == color_max) {
        h = value;
      }else{
        h = (value - color_min) / (color_max - color_min);
      }
      let hsv = d3.hsl((1-h)*240, 0.7, 0.5);
      return hsv+"";
    },
    get_hsv_color: function(values, color_min, color_max) {
      if(values.length > 0){
        const sum = values.reduce((v,x) => v+=x,0);
        const avg = sum / values.length;
        return this.interpolation_hsv(avg, color_min, color_max);
      }
    },
    draw_graph: function() {
      const self = this;
      const color_values = this.file_data[this.color_index];
      const color_min = Math.min.apply(null, color_values);
      const color_max = Math.max.apply(null, color_values);

      let formatCount = d3.format(",.0f");

      let svg = d3.select('#hist'+this.id);
      let margin = {top: 10, right: 4, bottom: 6, left: 4};
      let width = +svg.attr("width") - margin.left - margin.right;
      let height = +svg.attr("height") - margin.top - margin.bottom;

      svg.selectAll("*").remove();
      let g = svg.append("g");

      let x = d3.scaleLinear()
        .domain([color_min, color_max])
        .rangeRound([0, width]);

      let bins = d3.histogram()
        .domain(x.domain())
        .thresholds(d3.range(color_min, color_max, (color_max - color_min) / 24))
        (color_values);

      let y = d3.scaleLinear()
        .domain([0, d3.max(bins, function(d) { return d.length; })])
        .range([height, 0]);

      let bar = g.selectAll(".bar")
        .data(bins)
        .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });

      bar.append("rect")
        .attr("x", 1)
        .attr("width", function(d) { return x(d.x1) - x(d.x0) - 2; })
        .attr("height", function(d) { return height - y(d.length); })
        .attr("fill", function(d,i) { return self.get_hsv_color(d, color_min, color_max); });

      g.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
        .style("text-anchor", "middle")
        .style("font-size", "8px");
    }
  }
}
</script>

<style lang="scss" scoped>
.histogram {
}
</style>
