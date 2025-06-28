# Neural Network from Scratch
A simple feedforward neural network implementation in C++ for Fashion-MNIST classification.
I need to remake everything. It took a whole year to training an epoch.

## Features
- **Multilayer Perceptron**: Configurable hidden layers
- **Activation Functions**: Sigmoid, ReLU, Softmax
- **Backpropagation**: Gradient descent optimization
- **Batch Normalization**: Included but not integrated
- **Fashion-MNIST Support**: Built-in dataset reader

## Architecture
- **Input Layer**: 784 nodes (28×28 pixels)
- **Hidden Layer 1**: 128 nodes (Sigmoid activation)
- **Hidden Layer 2**: 64 nodes (Sigmoid activation)
- **Output Layer**: 10 nodes (Softmax activation)

## Quick Start
1. **Prepare Dataset**: Place Fashion-MNIST files in `fashion/` directory:
   - `train-images-idx3-ubyte`
   - `train-labels-idx1-ubyte`
   - `t10k-images-idx3-ubyte`
   - `t10k-labels-idx1-ubyte`
2. **Compile**:
   ```bash
   g++ -o neural_network main.cpp neural_network.cpp layer.cpp node.cpp
   ```
3. **Run**:
   ```bash
   ./neural_network
   ```

## Configuration
Edit `main.cpp` to modify:
- `LEARNING_RATE`: Default 0.1
- `NUM_EPOCH`: Default 5
- `HIDDEN_LAYER1_SIZE`: Default 128
- `HIDDEN_LAYER2_SIZE`: Default 64

## File Structure
- `main.cpp` - Entry point and training loop
- `neural_network.{h,cpp}` - Main network class
- `layer.{h,cpp}` - Layer implementation
- `node.{h,cpp}` - Individual neuron
- `activation_function.h` - Activation functions
- `read_mnist.h` - Dataset loader
- `batch_norm.h` - Batch normalization (experimental)

## Performance
Typical accuracy on Fashion-MNIST: ~85-90% after 5 epochs.

## Notes
- Uses uniform weight initialization (-0.5 to 0.5)
- Sigmoid activation in hidden layers
- Softmax output for classification
- Basic gradient descent (no momentum/Adam)
