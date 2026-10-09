
# Training data: (input, output) pairs
training_data = [
    (1.0, 5.0),
    (2.0, 8.0),
    (3.0, 11.0),
    (4.0, 14.0),
    (5.0, 17.0)
]


# Set initial model parameters
weight = 0.5
bias = 0

learning_rate = 0.01
steps = 1000


for step in range(steps):
    total_loss = 0.0
    total_gradient_wegiht = 0.0
    total_gradient_bias = 0.0
    num_samples = len(training_data)


    for x, y in training_data:

        y_predict = weight * x + bias

        error = y_predict - y

        loss = error ** 2
        total_loss += loss

        gradient_weight = 2 * x * error
        total_gradient_wegiht += gradient_weight

        gradient_bias = 2 * error
        total_gradient_bias += gradient_bias

    avg_loss = total_loss / num_samples
    avg_gradient_weight = total_gradient_wegiht / num_samples
    avg_gradient_bias = total_gradient_bias / num_samples

    weight = weight - learning_rate * avg_gradient_weight
    bias = bias - learning_rate * avg_gradient_bias

    if (step + 1) % 100 == 0:
        print(f"Step: {step + 1}, Weight: {weight:.4f}, Bias: {bias:.4f},Prediction: {y_predict:.4f}, Avg Loss: {avg_loss:.4f}")


print("\nTesting Model:")
test_inputs = [6.0, 7.0, 10.0, 20.0, 300, 120]

for x_test in test_inputs:
    y_test_pred = weight * x_test + bias
    print(f"Input: {x_test:.1f} -> Predicted Output: {y_test_pred:.4f}")





