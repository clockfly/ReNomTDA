<template>
  <div class="histogram">
    <svg :id="'hist'+id" :width="width" :height="svg_height" v-if="histdata"></svg>
  </div>
</template>

<script>
export default {
  name: "Histgram",
  props: ["id", "histdata"],
  data: function() {
    return {
      'width': 56,
      'margin': {
        'left': 0,
        'right': 0,
        'top': 6,
        'bottom': 6,
      },
      'bin_count': 12,
      'bin_height': 4,
    }
  },
  computed: {
    bins() {
      return d3.histogram()
        .domain(this.scale_y.domain())
        .thresholds(d3.range(this.min, this.max,(this.max-this.min)/this.bin_count))
        (this.histdata);
    },
    max() {
      return Math.max(...this.histdata);
    },
    min() {
      return Math.min(...this.histdata);
    },
    scale_x() {
      return d3.scaleLinear()
        .domain([0,d3.max(this.bins,function(d){return d.length;})])
        .range([0,this.width]);
    },
    scale_y() {
      return d3.scaleLinear()
        .domain([this.min, this.max])
        .rangeRound([this.bin_height*this.bin_count, 0]);
    },
    svg() {
      return d3.select('#hist'+this.id);
    },
    svg_height() {
      return this.bin_height * this.bin_count + this.margin.top + this.margin.bottom;
    }
  },
  watch: {
    histdata: function() {
      this.draw_graph();
    }
  },
  methods: {
    draw_rect: function(g) {
      const self = this;
      let bar = g.selectAll(".bar")
        .data(self.bins)
        .enter().append("g")
        .attr("class", "bar")
        .attr("transform", function(d) {
          return "translate(0," + self.scale_y(d.x0) + ")";
        });

      bar.append("rect")
        .attr("x", 0)
        .attr("width", function(d) {return self.scale_x(d.length);})
        .attr("height", function(d) {return self.scale_y(d.x0)-self.scale_y(d.x1);})
        .attr("fill", function(d) { return "#5c94aa"; });
    },
    draw_graph: function() {
      if(this.histdata) {
        this.svg.selectAll("*").remove();
        let g = this.svg.append("g")
          .attr("transform", function(d) {
          return "translate(0," + 2 + ")";
        });
        this.draw_rect(g);
      }
    }
  },
  mounted: function(){
    this.draw_graph();
  },
}
</script>

<style lang="scss">
.histogram {
  svg {
    border-left: 1px solid #ccc;
  }
}

</style>