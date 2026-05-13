#what will we build on?
# 1. Add a regularizer and see how that changes the performance of the model (ridge)
# 2. stochastic gradient descent + momentum instead of normal gradient descent (ADAM)
# 3. we'll do a classifier instead of a regression model, to explore other activation functions (e.g. softmax)
# 4. dropout and batch normalization (normalize outputs of one layer as they go into the next)
# 5. cross entropy loss instead of MSE

# we will train a classifier to learn some sort of strange spiral shape or something like that
# model architecture:
# 2 layers
# first layer takes input -> batch normalize -> ReLU -> second layer -> softmax -> output
# training: on any given iteration, we'll kill say 30% of neurons. apply ADAM and batching to train.
# loss function will have a L2 regularizer 