# Memory Augmented Neural Network

Given an utterance *x*, we want to encode it with a tradictional encoder (LSTM, Transformer, CNN) and cache the resulting latent representations into a differentiable external memory that can be accessed and extends the model capacity.

Attending memory keys during training leads to obtain positive and negative gradients according to whether the predicted value matches the input label *y* or not. Both gradients minimize the loss function and update the model with better traininable parameters, which generates better latent encodings to store in memory. 

![alt text](https://github.com/omar-florez/neural-augmented-neural-network/blob/master/figures/MANN.png)
