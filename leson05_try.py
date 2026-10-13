
inputs = [
    [2.0, 3.0],
    [4.0, 1.0],
    [1.0, 2.0],
    [3.0, 5.0],
    [2.0, 4.0],
    [5.0, 1.0],
    [3.0, 2.0]
]

weights = [ 0.5, 1.0]
bias = 0.2

def dot_product(matrix_a, matrix_b):
    result = 0.0

    for a, b in zip(matrix_a, matrix_b):
        result += a * b

    return result

def forward_pass(inputs, weights, bias):

    outputs = []
    for input in inputs :
        output = dot_product(input, weights) + bias

        outputs.append(output)
        
    return outputs
        
        
        
predictions = forward_pass(inputs, weights, bias)
        
print(f"Inputs: {inputs}")
print(f"Predictions: {predictions}")         