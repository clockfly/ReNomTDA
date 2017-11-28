import Vue from 'vue'
import Router from 'vue-router'
import TopologyPage from '../components/page/topology/Page.vue'
import DataSelectionPage from '../components/page/dataselection/Page.vue'

Vue.use(Router)

const router = new Router({
  routes: [
    { path: '/', name: 'DataSelectionPage', component: DataSelectionPage },
    { path: '/topology', name: 'TopologyPage', component: TopologyPage },
  ]
})

export default router