
# requirements:
numpy
tensorflow
phevaluator
++?

# How to run
open main and run the program

## Marameters
you can change the parameters in the config file, however make sure NN are trained with new parameters in mind

# Modules
## Game Manager
Creates the players, sets up the game, and controls the current players and flow of the game, together with deciding what legal actions a player can take

## Poker Oracle
Creates the decks, picking cards based on current cards on table, computes win probability using rollout which is used to compute the cheat sheet(which is not beeing used). Also more importantly it can rank any two collections of hands, currently using an extension phyevaluator, because my own was too slow. It uses this to create the utility matrix used in resolving.

## State Manager
creates and modifies states based on legal actions from the game manager and selection of the players or other components that need to change the states.

## Resolver
takes a state and generates a tree of nodes with different states, which is used to update ranges and evaluations that generates regret matrixes which is used to optimalize strategies. This is used for resolving and by the resolving AI agents. 

## NN manager
generates training data using the resolver. It bootstraps training where the utility matrixes are used to train generate the river data, and then the river model is used for the turn data etc...
### How to generate data and train models
go to the NN_creation file and run the program. Make sure lower tier models are created before creating the data. The sequence is (flop-turn-river)
(data generation takes a long time so the current models have only about 100 instances, however training data can be accumulated over time, simply generate more data and then a new model)

## Additional components
### Players- User and AI 
### Nodes- Chance and Action
### State


