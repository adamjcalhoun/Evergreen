

# what do we want to train the agent on?
env = WormWorldEnv(enable_render=True,world_size=(32,32))
# 1 odor plume

# should I add some generator for generating the environment?
env.add_odor_source(source_loc=(14,14))

# 5 odor plumes
for n in list(range(5)):
	env.add_odor_source(source_loc=(int(np.random.random(1)*32),int(np.random.random(1)*32)))

# 2 odor plumes, one decays through time
env.add_odor_source(source_loc=(14,14))
env.add_odor_source(source_loc=(14,14),decay_rate=0.1)

# temperature (wrong temp = negative utility)
env.add_temp_gradient(source_loc=(14,14))

# temperature + odor source
env.add_temp_gradient(source_loc=(14,14))
env.add_odor_source(source_loc=(18,14))

# odor plumes associated with food: only reward on food
env.add_odor_source(source_loc=(18,14),odor_type='food')
env.add_odor_circle(source_loc=(18,14),odor_type='food')

# odor plumes associated with disease: negative reward on reaching the peak
env.add_odor_source(source_loc=(18,14),odor_type='food')
env.add_odor_source(source_loc=(14,14),odor_type='disease')

# multi-agent
# each agent releases odor plumes
# -> this plus some limited amount of control

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
# normalization

# what do we want to examine?
# emergence of behaviors through training (do we first get hill-climbing, then eventually they learn to leave (forage) for new patches?)
# the computations the animals are performing

# better training:
# use keras-rl? 
# https://hub.packtpub.com/build-reinforcement-learning-agent-in-keras-tutorial/
# https://github.com/keras-rl/keras-rl/tree/master/examples
