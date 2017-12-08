<template>
  <div class="d3histogram">
    <svg :id="id" width="210" height="150"></svg>
  </div>
</template>

<script>
export default {
  name: "D3Histgram",
  props: ["id"],
  mounted: function(){
    this.draw_graph();
  },
  methods: {
    draw_graph: function() {
      let data_color_values;
      let statistic_value;
      if (this.id === "hist"){
        data_color_values = this.$store.state.topology.data_color_values;
        statistic_value = this.$store.state.topology.statistic_value;
      }else{
        data_color_values = this.$store.state.topology.data_color_values_col2;
        statistic_value = this.$store.state.topology.statistic_value_col2;
      }
      let formatCount = d3.format(",.0f");

      let margin = {top: 10, right: 30, bottom: 50, left: 50};
      let svg = d3.select('#'+this.id),
        width = +svg.attr("width") - margin.left - margin.right,
        height = +svg.attr("height") - margin.top - margin.bottom,
        g = svg.append("g").attr("transform", "translate(" + margin.left + "," + margin.top + ")");

      let min_x = Math.min.apply(null, data_color_values)
      let max_x = Math.max.apply(null, data_color_values)
      let x = d3.scaleLinear()
        .domain([min_x, max_x])
        .rangeRound([0, width]);

      let bins = d3.histogram()
        .domain(x.domain())
        .thresholds(d3.range(min_x, max_x, (max_x - min_x) / 20))
        (data_color_values);

      let y = d3.scaleLinear()
        .domain([0, d3.max(bins, function(d) { return d.length; })])
        .range([height, 0]);

      let interpolation_hsv = function(value, q_min, q_max, val_min, val_max) {
        let i;
        if(val_min == val_max){
          i = value;
        }else{
          i = (value - val_min) / (val_max - val_min);
        }
        let h = i * 0.25 + q_min;
        let hsv = d3.hsl((1-h)*240, 0.7, 0.5);
        return hsv+"";
      }

      let get_hsv_color = function(values) {
        let avg = values[0];

        if(avg < statistic_value[0]){
          return interpolation_hsv(avg, 0, 0.25, min_x, statistic_value[0]);
        }else if(avg >= statistic_value[0] && avg <= statistic_value[1]){
          return interpolation_hsv(avg, 0.25, 0.5, statistic_value[0], statistic_value[1]);
        }else if(avg >= statistic_value[1] && avg <= statistic_value[2]){
          return interpolation_hsv(avg, 0.5, 0.75, statistic_value[1], statistic_value[2]);
        }else{
          return interpolation_hsv(avg, 0.75, 1.0, statistic_value[2], max_x);
        }
      }

      let bar = g.selectAll(".bar")
        .data(bins)
        .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function(d) { return "translate(" + x(d.x0) + "," + y(d.length) + ")"; });

      bar.append("rect")
        .attr("x", 1)
        .attr("width", function(d) { return x(d.x1) - x(d.x0) - 2; })
        .attr("height", function(d) { return height - y(d.length); })
        .attr("fill", function(d) { return get_hsv_color(d); });

      g.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("dy", ".15em")
        .attr("transform", "rotate(-65)")
        .style("font-size", "9px");

      g.append("g")
        .call(d3.axisLeft(y))
        .selectAll("text")
        .style("text-anchor", "end")
        .style("font-size", "9px");;
    }
  }
}
</script>

<style lang="scss">
.d3histogram {
  position: absolute;
  top: 0;
  right: 0;
  width: 210px;
  height: 150px;
  svg {
    width: 210px;
    height: 150px;
  }
}

</style>