
global blinds 
global deck_size
global type_size
global chance_nodes
global all_in_dampener
global start_stack
global training_instances
global rollout_instances
global resolving_iterations

#global variables used over multiple files, make sure to update the NN training data if you change these

blinds= [20, 10]

deck_size = 24
type_size = int(deck_size / 4)

chance_nodes = 2

all_in_dampener = 10  #1-100

start_stack = 1000

training_instances = 1000

rollout_instances = 1000

resolving_iterations = 1