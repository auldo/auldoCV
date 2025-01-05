# About
In this project I collect my low-level machine learning and computer vision knowledge. It consists of a library containing the actual implementations and a small testing kit, as well as some examples. Of course, this implementation is not more than a fun project of mine to deepen my knowledge on DL architectures and basics. However, it may give some ideas of how to implement function approximating algorithms.

## Features
Currently, the project provides an implementation of the following features. It's important to note, that no other libraries than OpenCV are used and OpenCV is exclusively used to abstract away from specific file encodings. All other implementation has been done from scratch in modern CPP.
1. Modern CPP: Meaning, no raw pointers are used. Instead, the project profits from modern C++ like concepts and type traits. C++ 17 is used as core standard.
2. Vector implementation
3. Tensor implementation
4. Gradient implementation based on a Compute Graph implementation
5. Implementation of fully connected layers
6. Implementation of kernels
7. Implementation of convolutional layers
8. Implementation of DL optimizers:
   1. Stochastic Gradient Descent (SGD)
   2. Mini-Batch Gradient Descent
   3. Parallel Mini-Batch Gradient Descent