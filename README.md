# Memory Augmented Neural Network

Given an annotated utterance *(x, y)*, we encode *x* with an encoder (LSTM, Transformer) and cache similar latent representations generated during training into an external memory. Storage and retrieval operations are differentiable using attention over memory entries and extend encoder's capacity.

Attending the keys of the external memory module during training leads to obtain positive and negative gradients according to whether the predicted label associated to a memory entry matches the input label *y* or not. Both gradients minimize the loss function and update the model with memory cached traininable parameters. 

![alt text](https://github.com/omar-florez/neural-augmented-neural-network/blob/master/figures/MANN.png)
