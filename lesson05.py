# ================================================
# Training model with multiple inputs: Lesson 05
#=================================================


# ((input, input), target)
training_data = [
    ((1, 2), 3),
    ((4, 5), 6),
    ((7, 8), 9),
    ((3, 6), 7),
]

weight = [ 0.5, 1.0]
bias = 0.0
learning_rate = 0.01
steps = 1000
num_features = 2

print('/n Training in progress')

for steps in range(steps):
    total_bias_gradient = 0.00
    total_weight_gradient = [0.00] * num_features
    samples = len(training_data)

    for x_values, y in training_data:
        y_pred = sum(w * x for w, x in zip(weight, x_values)) + bias

        error = y_pred - y

        weight_gradient = [2 * x_num * error for x_num in x_values]
        for i in range(num_features):
            total_weight_gradient[i] += weight_gradient[i]

        bias_gradient = 2 * error
        total_bias_gradient += bias_gradient    

    avg_weight_gradient = [total_weight_gradient[i] / samples for i in range(num_features)]
    avg_bias_gradient = total_bias_gradient / samples
    for i in range(num_features):
        weight[i] = weight[i] - learning_rate * avg_weight_gradient[i]
        
    bias = bias - learning_rate * avg_bias_gradient

    if steps % 10 == 0:
        formatted_weights = [f"{w:.4f}" for w in weight]
        print(f"prediction:{y_pred:.4f}| Weights: {formatted_weights} | Bias: {bias:.4f}")


print("\n--- Testing Model ---")
test_inputs = [
    (6, 7),
    (10, 20),
    (0, 1)
]

for x_test in test_inputs:
    y_test_pred = sum(w * x for w, x in zip(weight, x_test)) + bias
    print(f"Input: {x_test} -> Predicted Output: {y_test_pred:.4f}")