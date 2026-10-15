import numpy as aj

inputs = aj.array([
    [4.0, 2.0],
    [2.0, 3.0],
    [5.0, 1.0],
])

weights = aj.array([
    [2.0, 3.0],
    [4.0, 1.0],
    [1.0, 5.0]
])

biases = aj.array([1.0, 2.0, 3.0])

def forward(a, b, c):

    weighted_sum = aj.dot(a, b.T)

    outputs = weighted_sum + c

    return outputs

predictions = forward(inputs, weights, biases)

print(f"Input shape: {inputs.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Biases shape: {biases.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions: {predictions}")

