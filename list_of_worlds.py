
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


########
# Order of things to add:
# - [x] Large environment (train [ ])
# - [x] Internal hunger for agent (train [ ])
# - [ ] Ability to lay eggs/get reward from this (train [ ])
# - [ ] Olfactory odors from agents (train with multiple agents [ ])
# - [ ] LSTM networks
# - [ ] Ability to control odors from agents (train [ ])
# - [ ] Ability to predate on other agents (train [ ])
# - [ ] Add cost for movement
########

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


# installing environment
# note that tensorflow2 is ~10x slower than tensorflow1. I think there is a parameter that can speed tf2 up
# but I have not yet tested that (add a line in the code that says '@tf.function')
conda create -n evergreen python=3
conda activate evergreen
conda install numpy tensorflow-gpu=1.15 git h5py
pip install gym
pip install pygame
git clone https://github.com/adamjcalhoun/Evergreen.git