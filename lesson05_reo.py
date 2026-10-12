# LESSON 5: Vectors, matrices and neural networks

# input vector 

inputs = [4.0, 2.0]

# Weights matrix

weights = [
    [2.0, 3.0],
    [4.0, 1.0],
    [1.0, 2.0],
    [3.0, 5.0],
    [2.0, 4.0],
    [5.0, 1.0],
    [3.0, 2.0]
]

biases = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def dot_product(vector_a, vector_b):

    result = 0.0

    for a, b in zip(vector_a, vector_b): # [(4, 2), (2,3)]
        result += a * b

    return result    


def forward(inputs, weights, biases):

    outputs = []

    for weight, bias in zip(weights, biases):
        output = dot_product(inputs, weight) + bias

        outputs.append(output)

    return outputs



predictions = forward(inputs, weights, biases)

print(f"Inputs: {inputs}")
print(f"Predictions: {predictions}")









# prediction1 = (w1x1+ w2x2) + b1

# prediction2 = (m1x1 + m2x2) + b2

# prediction3 = n1x1 + n2x2 + b3









zip()