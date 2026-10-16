# LESSON 7: PyTorch tensors

import torch

# inputs = torch.tensor([
#     [4.0, 2.0],
#     [2.0, 3.0],
#     [5.0, 1.0],
# ])


# weights = torch.tensor([
#     [2.0, 3.0],
#     [4.0, 1.0],
#     [1.0, 5.0],
# ])

# biases = torch.tensor([1.0, 2.0, 3.0])

# def forward(inputs, weights, biases):
#     # TODO 1:
#     # Multiply the input matrix by the
#     # transpose of the weight matrix.
#     weighted_sums = torch.matmul(inputs, weights.T)

#     # TODO 2:
#     # Add the bias vector to the results.
#     outputs = weighted_sums + biases

#     return outputs


# predictions = forward(inputs, weights, biases)

# print("inputs:")
# print(inputs)

# print("\nInput shape:", inputs.shape)
# print("Weight shape:", weights.shape)
# print("Bias shape:", biases.shape)


# print("Predictions:")
# print(predictions)

# print("\nOutput shape:", predictions.shape)


print("\n--- Automatic differentiation experiment ---")

x = torch.tensor(4.0)
y = torch.tensor(8.0)



w = torch.tensor(0.5, requires_grad=True)


prediction = w * x

loss = (prediction - y) ** 2

print("Prediction:", prediction.item())
print("Loss:", loss.item())

loss.backward()

print("Gradient calculated by PyTorch:", w.grad.item())