training_data= [
    (1, 5),
    (2, 8),
    (3, 11),
    (4, 14),
    (5, 17),
    (6, 20),
    (7, 23),
    (8, 26),
    (9, 29),
    (10, 32),
]

#parameters
weight = 0.5
bias= 0.0
learning_rate= 0.01
steps = 1000

for steps in range(steps):
    total_weight_gradient = 0.00
    total_bias_gradient = 0.00
    samples = len(training_data)

    for x, y in training_data:
        y_predict = weight * x + bias

        error = y_predict - y
        loss = error ** 2

        weight_gradient = 2 * x * error
        total_weight_gradient += weight_gradient

        bias_gradient = 2 * error
        total_bias_gradient += bias_gradient

    avg_weight_gradient = total_weight_gradient / samples
    avg_bias_gradient = total_bias_gradient / samples

    weight = weight - learning_rate * avg_weight_gradient
    bias = bias - learning_rate * avg_bias_gradient

    if steps % 10 == 0:
     print(f"prediction:{y_predict :.4f}, weight:{weight :.4f}, bias:{bias :.4f}")

print('/n testing model')
test_inputs = [
   6, 7, 10, 20, 300, 120, 0, 1, 2, 3, 4, 5, 8, 11, 12, 13, 14, 15
]

for x_test in test_inputs:
    y_test_pred = weight * x_test + bias
    print(f"Input: {x_test:.1f} -> Predicted Output: {y_test_pred:.4f}")
