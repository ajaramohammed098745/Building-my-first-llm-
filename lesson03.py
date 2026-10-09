training_data = [
    (1.0, 5.0),
    (2.0, 8.0),
    (3.0, 11.0),                     # y = w * x + b
    (4.0, 14.0),
    (5.0, 17.0)
]



weight = 0.5
learning_rate = 0.01
bias = 1
steps = 100

for steps in range(1, steps+2):
    total_loss = 0.00
    total_weight_gradient = 0.00
    total_bias_gradient = 0.00
    num_samples = len(training_data)

    for x, y, in training_data:
        y_pred = weight * x + bias

        loss = (y_pred - y)**2
        total_loss += loss

        gradient_weight = 2 * x * (weight * x + bias - y)
        total_weight_gradient += gradient_weight

        gradient_bias = 2 * (weight * x + bias -y)
        total_bias_gradient = gradient_bias

    avg_loss = total_loss / num_samples
    avg_weight_gradient = total_weight_gradient / num_samples
    avg_bias_gradient = total_bias_gradient / num_samples

    weight = weight - learning_rate * avg_weight_gradient
    bias = bias -learning_rate * avg_bias_gradient

    print(f"Weight:{weight:.2f}", f"loss:{avg_loss:.2f}", f"bias:{bias:.4f}")
    
    
print("\nTesting Model:")
test_inputs = [6.0, 7.0, 10.0, 20.0]

for x_test in test_inputs:
    # TODO G: Predict the output for the test inputs using your trained weight
    y_test_pred = weight * x_test
    print(f"Input: {x_test:.1f} -> Predicted Output: {y_test_pred:.4f}")
