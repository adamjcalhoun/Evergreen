
# WORLD 1.1
# -> Show we can chemotax
# WORLD 1.2
# -> Show we can thermotax
# WORLD 1.3
# -> Show we can learn chemotaxis based only on food association
# WORLD 1.4
# -> Show we can learn aversive chemotaxis based only on food association
# WORLD 1.5
# -> Show we can learn chemotaxis + thermotaxis

# WORLD 2.1
# -> Time-varying hunger (state-based behaviors)

# WORLD 3.1
# -> Patchy distribution of food (foraging?)

# WORLD 4.1
# -> Egg laying

# WORLD 5.1
# -> Multi-agent competition/collaboration

# BODY 1.1
# -> Multilayer dense network
# BODY 1.2
# -> Normalization/regularization
# BODY 1.3
# -> Vary number of layers and units/layer
# BODY 2.1
# -> LSTM
# BODY 3.1
# -> Feedback (recurrence) from lower layers


# has anyone shown a neural network or RL trained to move and to get reward based on food will perform chemotaxis?

# other resources: CO2, O2
# other internal needs: protein, sugars, salts, etc
# day/night or other environmental oscillations

# https://en.wikipedia.org/wiki/Marginal_value_theorem



########
# Order of things to add:
# - [x] Large environment (train [ ])
# - [x] Internal hunger for agent (train [ ])
# - [x] Ability to lay eggs/get reward from this (train [ ])
# - [x] Ability to use multiple agents in the environment
# -> Need to convert all hunger etc into lists...
# - [x] Olfactory odors from agents (train with multiple agents [ ])
# - [ ] LSTM networks
# -> http://digital-thinking.de/keras-returning-hidden-state-in-rnns/
# -> https://github.com/omerbsezer/LSTM_RNN_Tutorials_with_Demo
# - [ ] Ability to control odors from agents (train [ ])
# - [ ] Ability to predate on other agents (train [ ])
# - [ ] Add cost for movement
# - [ ] Add continuous action space
# - [ ] Add heterogeneous (multi-species) agent world
########

########
# What are figures/panels?
# 
# 1. An introduction to Evergreen
# 2. Behavioral: Show chemotaxis
# 3. Behavioral: Show foraging
# 4. Behavioral: Effects of other animals
# 5. Behavioral: Effects of internal hunger
# 6. Behavioral: Effects of needing multiple nutrients
# 7. Behavioral: Effects of predators
# 8. Internal: Representations with sensory bottleneck
# 9. Internal: Change in representations with social pressure
# 10. Internal: effects of feedback

########
# Want environments with many different resources

# how did that Zebrafish/worm paper train their models?



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
# what happens in networks as you bottleneck neurons (what are the representations)
# find examples of basic olfactory/etc behaviors in animals that we can find in these networks (a la DeepRetina)
# how do these change when in a social environment vs an isolated one?

# better training:
# use keras-rl? 
# https://hub.packtpub.com/build-reinforcement-learning-agent-in-keras-tutorial/
# https://github.com/keras-rl/keras-rl/tree/master/examples
# https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html

# LSTMs:
# https://www.memoria.ai/projects/revisiting-cartpole-experiments-with-lstms-part-1
# https://gist.github.com/giuseppebonaccorso/7040b10a13520c4b0340b8a89dc8262f
# https://github.com/keras-rl/keras-rl/issues/41
# https://github.com/keras-rl/keras-rl/issues/222


# installing environment
# note that tensorflow2 is ~10x slower than tensorflow1. I think there is a parameter that can speed tf2 up
# but I have not yet tested that (add a line in the code that says '@tf.function')
