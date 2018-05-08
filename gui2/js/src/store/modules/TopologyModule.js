import TopologyState from './state/TopologyState.js'
import TopologyGetter from './getter/TopologyGetter.js'
import TopologyMutation from './mutation/TopologyMutation.js'
import TopologyAction from './action/TopologyAction.js'

const TopologyModule = {
    state: TopologyState,
    getters: TopologyGetter,
    mutations: TopologyMutation,
    actions: TopologyAction
}

export default TopologyModule;