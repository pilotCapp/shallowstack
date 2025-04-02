import os
import sys
import numpy as np

from modules.state_modules.state import State
from modules.NN_modules.nn_manager import NN_manager 
from modules.resolver_modules.resolver import Resolver
import config


NN_manager = NN_manager(Resolver())


if __name__ == "__main__": # run this to select what you want to do
    choice = input("what do you want to do? (create data, train NN)")
    while choice not in ["create data", "train NN"]:
        print("please write a valid choice")
        choice = input("what do you want to do? (create data, train NN)")
    if choice == "create data":
        stage = input("what stage do you want to create data for (flop, turn, river)")
        while stage not in ["flop", "turn", "river"]:
            print("please write a valid stage")
            stage = input("what stage do you want to create data for (flop, turn, river)")
        overwrite = input("do you want to overwrite the existing data? (y,n)").lower()=="y"
        NN_manager.generate_training_data(stage, config.training_instances, overwrite)
        
    if choice == "train NN":
        stage = input("what stage do you want to train the NN for (flop, turn, river)")
        while stage not in ["flop", "turn", "river"]:
            print("please write a valid stage")
            stage = input("what stage do you want to train the NN for (flop, turn, river)")
        NN_manager.train_NN(stage)
    
    
