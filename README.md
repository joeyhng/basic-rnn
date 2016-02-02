#Basic RNN training example

Inherited from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn). This code is simplified from my other more complicated project, so it may contain unnecessary code.

As an example, the code process each mnist example as a sequence, where each timestep contains a image column as input feature. Therefore, each example contains 32 timesteps with 32 dimensional input.

To download mnist data used in the code:
```
wget https://s3.amazonaws.com/torch7/data/mnist.t7.tgz
tar xzf mnist.t7.tgz
```

