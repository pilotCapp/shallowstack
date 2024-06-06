import os
import sys
import numpy as np
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split

parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_parent_dir)

from modules.state_modules.state import State
from modules.oracle import Poker_Oracle
import config


class NN_manager:
    def __init__(self, resolver):
        self.resolver = resolver
        self.oracle = Poker_Oracle()

    def generate_training_data(self, type, training_instances=100, overwrite = False): #generates training data for selected part of the game
        if type =="flop":
            table_cards=3
        elif type =="turn":
            table_cards=4
        elif type =="river":
            table_cards=5
        else:
            raise ValueError("wrong number of cards in table")  
        

        if os.path.exists(f"data/{config.deck_size}/{type}.pkl") and not overwrite: #checks if data already exists and if it should be overwritten or not
            data = self.get_training_data(type)            
        else:
            data = []
            if not os.path.exists(f"data/{config.deck_size}/"):
                os.makedirs(f"data/{config.deck_size}/")
            with open(f"data/{config.deck_size}/{type}.pkl", "wb") as f:
                pickle.dump(data, f)
        print("creating training data for", type, "NN")

        for i in range(training_instances): #creates training data based on the number of instances wanted in the dataset
            table = self.oracle.pick_cards(np.array([]),table_cards)
            
            root_state = State()
            stake = np.random.randint(10, 100)*8

            root_state.pot = 2*stake
            root_state.current_bet = stake
            
            root_state.current_players = {
                "AI 1": [config.start_stack - stake, stake],
                "Player 1": [config.start_stack - stake, stake],
            }
            
            root_state.table = table
            root = self.resolver.init_node(
                np.full((config.deck_size, config.deck_size), 2 / config.deck_size**2),
                np.full((config.deck_size, config.deck_size), 2 / config.deck_size**2),
                "AI 1",
                root_state,
            )
            
            self.resolver.generate_subtree(root)
            self.resolver.range_update(root)
            self.resolver.evaluate_regret_strategize(root)

            instance = {
                "R1": root.R1,
                "R2": root.R2,
                "pot": root.state.pot,
                "table": root.state.table,
                "v1*": root.v1,
                "v2*": root.v2,
            }
            data.append(instance) #appends the instance to the data list after each instance
            with open(f"data/{config.deck_size}/{type}.pkl", "wb") as f:
                pickle.dump(data, f)
            print("instance", i, "created")
            

        print(f"Training data for {type} NN created and saved")

    def get_training_data(self, type): #loads training data from file
        if os.path.exists(f"data/{config.deck_size}/{type}.pkl"):
            with open(f"data/{config.deck_size}/{type}.pkl", "rb") as f:
                data = pickle.load(f)
            return data
        else:
            raise FileNotFoundError(f"No training data found of {type}")

    def train_NN(self, type, epochs=10, batch_size=32, overwrite=False): #trains the neural network based on the training data
        if os.path.exists(f"models/{config.deck_size}/{type}.h5") and not overwrite:
            raise FileExistsError(f"Model for {type} already exists")

        print("\ntraining started\n")

        data = self.get_training_data(type)

        input1_data = np.array([item["R1"] for item in data])
        input2_data = np.array([item["R2"] for item in data])
        input3_data = np.array([item["table"] for item in data])
        input4_data = np.array([item["pot"] for item in data])
        output1_data = np.array([item["v1*"] for item in data])
        output2_data = np.array([item["v2*"] for item in data])

        output1_data = output1_data.reshape((len(output1_data), -1))
        output2_data = output2_data.reshape((len(output2_data), -1))

        #splits the data into training and testing data
        input1_train, input1_test, input2_train, input2_test, input3_train, input3_test, input4_train, input4_test, output1_train, output1_test, output2_train, output2_test = train_test_split(
        input1_data, input2_data, input3_data, input4_data, output1_data, output2_data, test_size=0.2, random_state=42
    )

        # Defines the input layers
        input1 = keras.Input(shape=(config.deck_size, config.deck_size))
        input2 = keras.Input(shape=(config.deck_size, config.deck_size))
        input3 = keras.Input(shape=(self.get_type_int(type),))
        input4 = keras.Input(shape=(1,))

        # Flattens the inputs because the model only accepts 1D inputs and vectors are 2D
        flat1 = keras.layers.Flatten()(input1)
        flat2 = keras.layers.Flatten()(input2)
        flat3 = keras.layers.Flatten()(input3)
        flat4 = keras.layers.Flatten()(input4)

        # Concatenates all inputs
        concat = keras.layers.concatenate([flat1, flat2, input3, input4])

        max_neurons= int(config.deck_size**2/2) #selects neurons based on the size of the deck

        dense1 = keras.layers.Dense(max_neurons, activation="relu")(concat)
        dense2 = keras.layers.Dense(max_neurons/2, activation="relu")(dense1)
        dense3 = keras.layers.Dense(max_neurons/4, activation="relu")(dense2)
        dense4 = keras.layers.Dense(max_neurons/8, activation="relu")(dense3)

        output1 = keras.layers.Dense(config.deck_size * config.deck_size, activation="linear")(dense4)
        output2 = keras.layers.Dense(config.deck_size * config.deck_size, activation="linear")(dense4)

        model = keras.Model(
            inputs=[input1, input2, input3, input4], outputs=[output1, output2]
        )

        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit( #trains the model
            [input1_train, input2_train, input3_train, input4_train],
            [output1_train, output2_train],
            epochs=epochs,
            batch_size=batch_size,
        )
        
        loss = model.evaluate( #evaluates the model
        [input1_test, input2_test, input3_test, input4_test],
        [output1_test, output2_test],
        verbose=0
    )

        # Save the model to a file
        model.save(f"models/{config.deck_size}/{type}.h5")
        print(f"model {type} created with total mse over both vectors {loss[0]}")

    def get_model(self, type): #loads the model from file
        if os.path.exists(f"models/{config.deck_size}/{type}.h5"):
            model = keras.models.load_model(f"models/{config.deck_size}/{type}.h5")
            return model
        else:
            raise FileNotFoundError(f"No model found of {type}")

    def predict(self, r1, r2, table, pot): #predicts the value of the cards based on the model and the input data
        type
        model = self.get_model(self.get_type(table))
        r1 = np.expand_dims(r1, axis=0)
        r2 = np.expand_dims(r2, axis=0)
        table = np.expand_dims(table, axis=0)
        pot = np.expand_dims(np.array([pot]), axis=0)

        v1_, v2_ = model.predict([r1, r2, table, pot],verbose=0)
        v1_2D = np.reshape(v1_[0], (config.deck_size, config.deck_size))
        v2_2D = np.reshape(v2_[0], (config.deck_size, config.deck_size))

        return v1_2D, v2_2D

    def get_type(self, table): #gets the type of the table based on the number of cards
        if len(table) == 3:
            type = "flop"
        elif len(table) == 4:
            type = "turn"
        elif len(table) == 5:
            type = "river"
        else:
            raise ValueError("wrong number of cards in table")
        return type
    
    def get_type_int(self,type): #gets the type number of cards on table based on type
        return {
            "flop": 3,
            "turn": 4,
            "river": 5,
        }[type]


