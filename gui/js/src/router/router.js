import Vue from 'vue'
import Router from 'vue-router'
import TopologyPage from '../components/page/topology/page.vue'
import DataSelectionPage from '@/components/page/data_selection/page.vue'

Vue.use(Router)

const router = new Router({
  routes: [
    { path: '/', name: 'DataSelectionPage', component: DataSelectionPage },
    { path: '/topology', name: 'TopologyPage', component: TopologyPage },
  ]
})

export default router
