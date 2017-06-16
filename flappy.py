import random
import numpy as np
from evostra import EvolutionStrategy
from keras.layers import Flatten, Dense
from keras.models import Model, Input
from keras.optimizers import Adam
from ple import PLE
from ple.games.flappybird import FlappyBird


class Agent:

    AGENT_HISTORY_LENGTH = 4
    NUM_OF_ACTIONS = 2
    POPULATION_SIZE = 16
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.001

    
    def __init__(self):
        np.random.seed(0)
        self.model = self.get_model()
        self.game = FlappyBird(pipe_gap=120)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.env.init()
        self.env.getGameState = self.game.getGameState
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)


    def get_predicted_action(self, sequence):
        prediction = self.model.predict(np.asarray([sequence]))[0]
        x = np.argmax(prediction)
        return 119 if x == 1 else None
    
    
    def load(self, path='one_dense_64.h5'):
        self.model.load_weights(path)
        self.es.weights = self.model.get_weights()


    def get_observation(self):
        return np.array(self.env.getGameState().values())

    
    def play(self, episodes):
        self.env.display_screen = True
        self.model.set_weights(self.es.weights)
        for episode in xrange(episodes):
            self.env.reset_game()
            observation = self.get_observation()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            score = 0
            while not done:
                action = self.get_predicted_action(sequence)
                reward = self.env.act(action)
                observation = self.get_observation()
                sequence = sequence[1:]
                sequence.append(observation)
                done = self.env.game_over()
                if self.game.getScore() > score:
                    score = self.game.getScore()
                    print "score: %d" % score
        self.env.display_screen = False

    
    def train(self, iterations):
        self.es.run(iterations, print_step=1)


    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in xrange(self.EPS_AVG):
            self.env.reset_game()
            observation = self.get_observation()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                action = self.get_predicted_action(sequence)
                reward = self.env.act(action)
                if reward > 0:
                    reward = 100
                elif reward < 0:
                    reward = -1
                reward += random.choice([-0.01, 0.01])
                total_reward += reward
                observation = self.get_observation()
                sequence = sequence[1:]
                sequence.append(observation)
                done = self.env.game_over()

        return total_reward/self.EPS_AVG


    def get_model(self):
        input_layer = Input(shape=(self.AGENT_HISTORY_LENGTH, 8))
        layer = Flatten()(input_layer)
        layer = Dense(64)(layer)
        output_layer = Dense(self.NUM_OF_ACTIONS)(layer)
        model = Model(input_layer, output_layer)
        model.compile(Adam(self.LEARNING_RATE), 'mse')
        return model
