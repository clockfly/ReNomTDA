<template>
  <div :id="'canvas'+id"></div>
</template>

<script>
export default {
  name: "D3Canvas",
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
    colors() {
      return this.$store.state.topologies[this.id].colors[this.topology.color_index];
    },
    topology() {
      return this.$store.state.topologies[this.id];
    }
  },
  watch: {
    color_index: function() {
      this.reset_canvas();
      this.draw_graph();
    },
    colors: function() {
      this.reset_canvas();
      this.draw_graph();
    }
  },
  mounted: function() {
    this.draw_graph();
  },
  methods: {
    reset_canvas: function() {
      let e = d3.select('#canvas'+this.id).html("");
    },
    draw_graph: function() {
      const self = this;
      const nodes = this.topology.nodes;
      const sizes = this.topology.node_sizes;
      const edges = this.topology.edges;
      const colors = this.colors;
      const train_index = this.topology.train_index;

      const store = this.$store;

      const size_max = Math.max.apply(null, this.sizes);

      // selected spring or not
      const spring = this.$store.state.visualize_mode;

      // selected node
      let active_node_array = [];

      let links = []
      for(let i=0;i<edges.length;i++){
        links.push({source: edges[i][0], target: edges[i][1]})
      }

      let e = d3.select('#canvas'+this.id);
      const width = e._groups[0][0].clientWidth;
      const height = e._groups[0][0].clientHeight;

      let svg = e.append('svg')
        .attr('width', width)
        .attr('height', height);

      svg.append("g")
        .attr("class", "links")

      svg.append("g")
        .attr("class", "nodes")

      if(spring > 0 && links.length > 0){
        var simulation = d3.forceSimulation(nodes)
          .force('charge', d3.forceManyBody().strength(0))
          .force('collision', d3.forceCollide().radius(function(d,i) { return sizes[i]; }))
          .force('x', d3.forceX().x(function(d,i) {
            return nodes[i][0]*width;
          }))
          .force('y', d3.forceY().y(function(d,i) {
            return nodes[i][1]*height;
          }))
          .force('link', d3.forceLink(links)
          .distance(30)
          .strength(0.5))
          .on('tick', ticked);
      }else{
        let link = svg.selectAll("link")
          .data(edges, function(d) { return d[1]; })
          .enter().append("line")
          .attr("x1", function(d) { return nodes[d[0]][0]*width; })
          .attr("y1", function(d) { return nodes[d[0]][1]*height; })
          .attr("x2", function(d) { return nodes[d[1]][0]*width; })
          .attr("y2", function(d) { return nodes[d[1]][1]*height; })
          .style("stroke", function(d) {return colors[d[0]]; });

        let circles = svg.selectAll("circle")
          .data(nodes)
          .enter().append("circle")
          .attr("cx", function(d) {return (d[0]*width); })
          .attr("cy", function(d) {return (d[1]*height); })
          .attr("r", function(d,i) { return sizes[i]; })
          .attr("index", function(d,i) {return i; })
          .style("fill", function(d,i) {return colors[i]; })
          .style("opacity", function(d,i) {
            if(train_index.includes(i)){
              return 0.3;
            }else{
              return 1;
            }
          })
          .on('click', function(d, i){
            if(active_node_array.includes(i)) {
              d3.select(this).attr("stroke-width", function(d){ return 0; });
              active_node_array = active_node_array.filter(n => n != i);
              store.commit('remove_click_node', {
                'click_node_index': i,
                'index': self.id
              });
            }else{
              let n = d3.select(this);
              active_node_array.push(i);
              n.attr("stroke", function(d) { return d3.rgb(153,153,153,0.3); })
               .attr("stroke-width", function(d){ return 3; });
              store.commit('set_click_node', {
                'click_node_index': i,
                'index': self.id
              });
            }
          });
      }

      function updateLinks() {
        let u = svg.select(".links")
          .selectAll('line')
          .data(links)

        u.enter()
          .append('line')
          .merge(u)
          .attr('x1', function(d,i) { return d.source.x; })
          .attr('y1', function(d,i) { return d.source.y; })
          .attr('x2', function(d,i) { return d.target.x; })
          .attr('y2', function(d,i) { return d.target.y; })
          .attr('stroke', function(d,i){ return colors[edges[i][0]]; })

        u.exit().remove()
      }

      function updateNodes() {
        let u = svg.select(".nodes")
          .selectAll('circle')
          .data(nodes)

        u.enter()
          .append('circle')
          .merge(u)
          .attr("cx", function(d, i) { return d.x; })
          .attr("cy", function(d, i) { return d.y; })
          .attr('r', function(d,i) { return sizes[i]; })
          .attr('fill', function(d,i){ return colors[i]; })
          .style("opacity", function(d,i) {
            if(train_index.includes(i)){
              return 0.3;
            }else{
              return 1;
            }
          })
          .on('click', function(d, i){
            if(active_node_array.includes(i)) {
              d3.select(this).attr("stroke-width", function(d){ return 0; });
              active_node_array = active_node_array.filter(n => n != i);
              store.commit('remove_click_node', {
                'click_node_index': i,
                'index': self.id
              });
            }else{
              let n = d3.select(this);
              active_node_array.push(i);
              n.attr("stroke", function(d) { return d3.rgb(153,153,153,0.3); })
               .attr("stroke-width", function(d){ return 3; });
              store.commit('set_click_node', {
                'click_node_index': i,
                'index': self.id
              });
            }
          })
          .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

        u.exit().remove()
      }

      function ticked() {
        updateLinks()
        updateNodes()
      }
      function dragstarted(d) {
        if (!d3.event.active) simulation.alphaTarget(1).restart();
        d.fx = d.x;
        d.fy = d.y;
      }

      function dragged(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
      }

      function dragended(d) {
        if (!d3.event.active) simulation.alphaTarget(1);
        d.fx = null;
        d.fy = null;
      }
    }
  }
}
</script>

<style lang="scss">
#canvas0, #canvas1{
  width: 100%;
  height: 100%;
}
</style>
