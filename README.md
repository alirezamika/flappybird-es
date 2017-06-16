# Learning to play Flappy Bird using Evolution Strategies

This agent uses Evolutional Strategies and deep learning models to master the Flappy Bird game.

Read [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://blog.openai.com/evolution-strategies/) from OpenAI if you are interested.

After ~5000 iterations, it won't lose!


![demo](http://m.UploadEdit.com/ba3s/1497637053928.gif)

# Dependencies

- [tensorflow](https://www.tensorflow.org/)
- [keras](https://github.com/fchollet/keras)
- [evostra](https://github.com/alirezamika/evostra)
- [PyGame-Learning-Environment](https://github.com/alirezamika/PyGame-Learning-Environment)


# Usage

To see the agent playing the game:

```
from flappy import *

agent = Agent()

# the pre-trained weights are saved into 'one_dense_64.h5' which you can use.
agent.load('one_dense_64.h5')

# play one episode
agent.play(1)
```

To start training the agent:

```
# train for 100 iterations
agent.train(100)
```
