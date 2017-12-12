import Vue from 'vue'
import Vuex from 'vuex'
import TopologyModule from './modules/TopologyModule.js'

Vue.use(Vuex)

const store = new Vuex.Store({
    // TODO
    // Storeのアーキテクチャを変更する.
    modules: {
        topology: TopologyModule,
    }
})

export default store
