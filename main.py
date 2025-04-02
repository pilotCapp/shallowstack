import os
import sys
from modules.game_manager import Game_Manager
import config as config

if __name__ == "__main__":
    users = input("how many human players? (0-6)")
    while users.isdigit() == False:
        print("please write a valid number")
        users = input("how many human players?")
        
    bots = input("how many bots? (0-6)")
    while  bots.isdigit() == False:
        print("please write a valid number")
        bots = input("how many human players?")
            
    
    game_manager= Game_Manager(int(users),int(bots), config.blinds, config.rollout_instances)
    game_manager.start_game()

 
