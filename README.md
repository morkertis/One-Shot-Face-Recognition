# One-Shot-Face-Recognition
One-Shot Face Recognition Using Siamese Neural Networks

In this work, we used convolutional neural networks (CNNs) to carry out the task of facial recognition. 
More specifically, we have implemented a one-shot classification solution. 
In order to achieve this goal, we implemented a Siamese Neural Network(SNN), according to the paper:
[Siamese Neural Networks for One-shot Image recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
Our main goal was to successfully execute a one-shot learning task for previously unseen objects.
Given two facial image of previously unseen persons, our SNN will try and determine whether they are the same person.

Sklearn API:
In order to use the dataset with sklearn, we have changed the function fetch_lfw_pairs: 
1.	Change the function that will support one channel picture 
  - the change in sklearn\datasets\lfw.py --> comment added code
2.	Redirector / Change to our [dataset LFW-a](http://vis-www.cs.umass.edu/lfw/index.html#views) instead of the existing one, and not download the matching dataset (LFW).

[siamese neural networks](https://github.com/morkertis/One-Shot-Face-Recognition/blob/master/figures/siamese%20neural%20networks_v2.jpg)


references:
1. [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
2. [Contrastive Loss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)
