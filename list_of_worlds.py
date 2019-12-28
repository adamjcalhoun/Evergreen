

# what do we want to train the agent on?
# 1 odor plume
# 2 odor plumes
# 5 odor plumes
# 2 odor plumes, one decays through time
#
# temperature (wrong temp = negative utility)
#
# odor plumes associated with food: only reward on food
# odor plumes associated with disease: negative reward on reaching the peak
#
# multi-agent
# each agent releases odor plumes
# -> this plus some limited amount of control
# 
# predators/competitors
# herbivores, omnivores, carnivores
# reward if your species/team lays more healthy eggs


# what are the value functions that we want to train on?
# maximize odor?
# maximize food? food may all be eaten at once or can have depletion timescale
# balance of eggs on food patches that are rewarded N steps later + food eaten


# what different architectures do we want to use?
# simple: (multilayer) dense networks
# more complex: LSTMs
# feedback vs feed-forward only

# what do we want to examine?
# emergence of behaviors through training (do we first get hill-climbing, then eventually they learn to leave (forage) for new patches?)
# the computations the animals are performing

env = WormWorldEnv(enable_render=True,world_size=(32,32))