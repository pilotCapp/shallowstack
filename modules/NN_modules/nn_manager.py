import os
import sys
import numpy as np
import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow as tf


from modules.state_modules.state import State
from modules.oracle import Poker_Oracle
import config
from filelock import FileLock



class NN_manager:
    def __init__(self, resolver):
        self.resolver = resolver
        self.oracle = Poker_Oracle()
        self.models = {"flop":self.get_model("flop"), "turn":self.get_model("turn"),"river":self.get_model("river")} #TODO: fix bug which might allways require a model, even when only generating river training data, This is currently hard coded needing changing for each step of data generation
        self.lock = FileLock(str(config.REPO_ROOT) +"/lock" +".lock")

    def generate_training_data(self, stage_type, training_instances=100, overwrite = False, save_batch = 100): #generates training data for selected part of the game
        if stage_type =="flop":
            table_cards=3
            if self.models["flop"] == None:
                print("Error retrieving flop model")
        elif stage_type =="turn":
            table_cards=4
            if self.models["turn"] == None:
                print("Error retrieving turn model")
        elif stage_type =="river":
            if self.models["river"] == None:
                print("Error retrieving turn model")
            table_cards=5
        else:
            raise ValueError("wrong number of cards in table")  
        
        if not os.path.exists(f"data/{config.deck_size}/{stage_type}.pkl") or overwrite: #checks if data already exists and if it should be overwritten or not
                data = []
                if not os.path.exists(f"data/{config.deck_size}/"):
                    os.makedirs(f"data/{config.deck_size}/")
                with self.lock:
                    with open(f"data/{config.deck_size}/{stage_type}.pkl", "wb") as f:
                        pickle.dump(data, f)

        print("creating training data for", stage_type, "NN")
        batch=[]

        for i in range(training_instances): #creates training data based on the number of instances wanted in the dataset
            
            table = self.oracle.pick_cards(np.array([]),table_cards)            
            root_state = State()
            stake = np.random.randint(80, 800)

            root_state.pot = 2*stake
            root_state.current_bet = stake
            
            root_state.current_players = {
                "AI 1": [config.start_stack - stake, stake],
                "Player 1": [config.start_stack - stake, stake],
            }
            
            root_state.table = table
            node_init= 2 / config.deck_size**2
            root = self.resolver.init_node(
                np.full((config.deck_size, config.deck_size), node_init),
                np.full((config.deck_size, config.deck_size), node_init),
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
            batch.append(instance)

            if len(batch) % save_batch == 0:
                with self.lock:
                    with open(f"data/{config.deck_size}/{stage_type}.pkl", "rb+") as f:
                        data = pickle.load(f)
                        data.extend(batch)
                        data_length=len(data)
                        f.seek(0)
                        pickle.dump(data, f)
                        f.truncate()
                batch=[]
                print("current instances", i-save_batch+1, " to ", i+1, " created and saved with data length: ", data_length)
            else:
                print("batch size is: ", len(batch))
            

        print(f"Training data for {stage_type} NN created and saved")

    def get_training_data(self, stage_type): #loads training data from file
        if os.path.exists(f"data/{config.deck_size}/{stage_type}.pkl"):
            with self.lock:
                with open(f"data/{config.deck_size}/{stage_type}.pkl", "rb") as f:
                    data = pickle.load(f)
                return data
        else:
            raise FileNotFoundError(f"No training data found of {stage_type}")

    def train_NN(self, stage_type, epochs=100, batch_size=32, overwrite=False): #trains the neural network based on the training data
        if os.path.exists(f"models/{config.deck_size}/{stage_type}.keras") and not overwrite:
            raise FileExistsError(f"Model for {stage_type} already exists")

        print("\ntraining started\n")

        data = self.get_training_data(stage_type)
        print("training data length: ", len(data))

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
        input3 = keras.Input(shape=(self.get_stage_type_int(stage_type),))
        input4 = keras.Input(shape=(1,))

        # Flattens the inputs because the model only accepts 1D inputs and vectors are 2D
        flat1 = keras.layers.Flatten()(input1)
        flat2 = keras.layers.Flatten()(input2)
        flat3 = keras.layers.Flatten()(input3)
        flat4 = keras.layers.Flatten()(input4)

        # Concatenates all inputs
        concat = keras.layers.concatenate([flat1, flat2, input3, input4])

        max_neurons= int(config.deck_size**2/2) #selects neurons based on the size of the deck

        dense1 = keras.layers.Dense(int(max_neurons), activation="relu")(concat)
        dense2 = keras.layers.Dense(int(max_neurons/2), activation="relu")(dense1)
        dense3 = keras.layers.Dense(int(max_neurons/4), activation="relu")(dense2)
        dense4 = keras.layers.Dense(int(max_neurons/8), activation="relu")(dense3)

        output1 = keras.layers.Dense(int(config.deck_size * config.deck_size), activation="linear")(dense4)
        output2 = keras.layers.Dense(int(config.deck_size * config.deck_size), activation="linear")(dense4)

        model = keras.Model(
            inputs=[input1, input2, input3, input4], outputs=[output1, output2]
        )

        model.compile(optimizer="adam", loss="mean_squared_error")

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Metric to monitor
            patience=8,          # Number of epochs with no improvement after which training stops
            restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
        )

        model.fit( #trains the model
            [input1_train, input2_train, input3_train, input4_train],
            [output1_train, output2_train],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([input1_test, input2_test, input3_test, input4_test], [output1_test, output2_test]),
            callbacks=[early_stopping]
        )
        
        loss = model.evaluate( #evaluates the model
        [input1_test, input2_test, input3_test, input4_test],
        [output1_test, output2_test],
        verbose=0
    )

        # Save the model to a file
        model.save(f"models/{config.deck_size}/{stage_type}.keras")
        print(f"model {stage_type} created with total mse over both vectors {loss[0]}")

    def get_model(self, stage_type): #loads the model from file
        if os.path.exists(f"models/{config.deck_size}/{stage_type}.keras"):
            model = keras.models.load_model(f"models/{config.deck_size}/{stage_type}.keras")
            return model
        else:
            # raise FileNotFoundError(f"No model found of {stage_type}")
            return None

    @tf.function
    def predict(self, r1, r2, table, pot):  # predicts the value of the cards based on the model and the input data
        r1_tensor = tf.expand_dims(r1, axis=0)
        r2_tensor = tf.expand_dims(r2, axis=0)
        table_tensor = tf.expand_dims(table, axis=0)
        pot_tensor = tf.expand_dims(tf.convert_to_tensor([pot], dtype=tf.float32), axis=0)
        
        v1_, v2_ = self.models[self.get_stage_type(table)]([r1_tensor, r2_tensor, table_tensor, pot_tensor], training=False)
        v1_2D = tf.reshape(v1_[0], (config.deck_size, config.deck_size))
        v2_2D = tf.reshape(v2_[0], (config.deck_size, config.deck_size))
        return v1_2D, v2_2D

    def get_stage_type(self, table): #gets the stage_type of the table based on the number of cards
        if len(table) == 0:
            stage_type = "pre-flop"
        elif len(table) == 3:
            stage_type = "flop"
        elif len(table) == 4:
            stage_type = "turn"
        elif len(table) == 5:
            stage_type = "river"
            #raise ValueError("Should not be using river model for prediction?")
        else:
            raise ValueError("wrong number of cards in table, with table: ", table)
        return stage_type
    
    def get_stage_type_int(self,stage_type): #gets the stage_type number of cards on table based on stage_type
        return {
            "pre-flop": 0,
            "flop": 3,
            "turn": 4,
            "river":5
        }[stage_type]


