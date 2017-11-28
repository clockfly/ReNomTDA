<template>
  <div :id="id">
    <d3histgram v-if="show_histogram" :id="histid"></d3histgram>
  </div>
</template>

<script>
import D3Histgram from './D3Histgram.vue'

export default {
    name: "D3Canvas",
    props: ["id", "width", "height", "nodes", "edges", "colors", "sizes", "trainindex", "histid", "columns", "resolution"],
    components: {
        'd3histgram': D3Histgram
    },
    computed: {
        show_histogram() {
            return this.$store.state.topology.show_histogram;
        },
        show_spring() {
            return this.$store.state.topology.show_spring;
        }
    },
    mounted: function(){
        this.draw_graph();
    },
    methods: {
        draw_graph: function() {
            const width = this.width;
            const height = this.height;
            const nodes = this.nodes;
            const edges = this.edges;
            const colors = this.colors;
            const sizes = this.sizes;
            const train_index = this.trainindex;
            const store = this.$store;
            const columns = this.columns;
            const resolution = this.resolution;

            const size_max = Math.max.apply(null, this.sizes);

            let active_node = undefined;

            let links = []
            for(let i=0;i<edges.length;i++){
                links.push({source: edges[i][0], target: edges[i][1]})
            }

            let svg = d3.select('#'+this.id).append('svg')
                .attr('width', width)
                .attr('height', height);

            svg.append("g")
                .attr("class", "links")

            svg.append("g")
                .attr("class", "nodes")

            if(this.show_spring && links.length > 0){
                var simulation = d3.forceSimulation(nodes)
                .force('charge', d3.forceManyBody().strength(-5))
                .force('collision', d3.forceCollide().radius(function(d,i) { return sizes[i]*12; }))
                .force('x', d3.forceX().x(function(d,i) {
                    return setCoordinate(nodes[i][0], width);
                }))
                .force('y', d3.forceY().y(function(d,i) {
                    return setCoordinate(nodes[i][1], height);
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
                    .style("stroke", function(d) {return colors[d[0]] });

                let circles = svg.selectAll("circle")
                    .data(nodes)
                    .enter().append("circle")
                    .attr("cx", function(d) {return (d[0]*width); })
                    .attr("cy", function(d) {return (d[1]*height); })
                    .attr("r", function(d,i) { return sizes[i]*10; })
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
                        if(active_node) {
                            active_node.attr("stroke-width", function(d){ return 0; });
                        }
                        active_node = d3.select(this);
                        active_node.attr("stroke", function(d) { return d3.rgb(153,153,153,0.3); })
                                   .attr("stroke-width", function(d){ return 3; });
                        store.dispatch('click_node', {
                            'click_node_index': i,
                            'columns': columns
                        });
                    });
            }

            function setCoordinate(data, bound){
                if(data < 0.2){
                    return bound*0.1;
                }else if(data >= 0.2 && data < 0.4){
                    return bound*0.3
                }else if(data >= 0.4 && data < 0.6){
                    return bound*0.5;
                }else if(data >= 0.6 && data < 0.8){
                    return bound*0.7;
                }else{
                    return bound*0.9;
                }
            }

            function checkBoundary(data, r, bound){
                if((data+r) > bound){
                    return bound - r;
                }else if((data-r) < 0){
                    return r;
                }else{
                    return data;
                }
            }

            function updateLinks() {
                let u = svg.select(".links")
                    .selectAll('line')
                    .data(links)
                u.enter()
                    .append('line')
                    .merge(u)
                    .attr('x1', function(d,i) {
                        let r = sizes[edges[i][0]]*10;
                        return d.source.x = checkBoundary(d.source.x, r, width);
                    })
                    .attr('y1', function(d,i) {
                        let r = sizes[edges[i][0]]*10;
                        return d.source.y = checkBoundary(d.source.y, r, height);
                        
                    })
                    .attr('x2', function(d,i) {
                        let r = sizes[edges[i][1]]*10;
                        return d.target.x = checkBoundary(d.target.x, r, width);
                    })
                    .attr('y2', function(d,i) {
                        let r = sizes[edges[i][1]]*10;
                        return d.target.y = checkBoundary(d.target.y, r, height);
                    })
                    .attr('stroke', function(d,i){return colors[edges[i][0]];})
                u.exit().remove()
            }

            function updateNodes() {
                let u = svg.select(".nodes")
                    .selectAll('circle')
                    .data(nodes)
                u.enter()
                    .append('circle')
                    .merge(u)
                    .attr("cx", function(d, i) {
                        let r = sizes[i]*10;
                        return d.x = checkBoundary(d.x, r, width);
                    })
                    .attr("cy", function(d, i) {
                        let r = sizes[i]*10;
                        return d.y = checkBoundary(d.y, r, height);
                    })
                    .attr('r', function(d,i) { return sizes[i]*10; })
                    .attr('fill', function(d,i){ return colors[i]; })
                    .style("opacity", function(d,i) {
                        if(train_index.includes(i)){
                            return 0.3;
                        }else{
                            return 1;
                        }
                    })
                    .on('click', function(d, i){ 
                        if(active_node) {
                            active_node.attr("stroke-width", function(d){ return 0; });
                        }
                        active_node = d3.select(this);
                        active_node.attr("stroke", function(d) { return d3.rgb(153,153,153,0.3); })
                                   .attr("stroke-width", function(d){ return 3; })
                                   .attr();
                        store.dispatch('click_node', {
                            'click_node_index': i,
                            'columns': columns
                        });
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
#canvas, #canvas2{
    position: relative;
}
</style>