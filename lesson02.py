# =====================================================================
# LESSON 2 ASSIGNMENT: Training on Multiple Examples
# =====================================================================
# Objective: Train a single-parameter model (y = weight * x) on a 
# dataset of multiple examples using Batch Gradient Descent.

# 1. Training dataset (input, target)
training_data = [
    (1.0, 2.0),
    (2.0, 4.0),
    (3.0, 6.0),
    (4.0, 8.0),
    (5.0, 10.0),
]

# 2. Initial model parameters
weight = 0.0
learning_rate = 0.01
steps = 30

print("Starting Training...\n")

# 3. Training loop
for step in range(1, steps + 1):
    total_loss = 0.0
    total_gradient = 0.0
    n_samples = len(training_data)
    
    # Iterate through each example in the dataset
    for x, y_true in training_data:
        
        # TODO A: Predict the output using the current weight
        y_pred = weight * x
        
        # TODO B: Calculate the Squared Error Loss for this example
        loss = (y_pred - y_true)**2
        total_loss += loss
        
        # TODO C: Calculate the Gradient (derivative of loss w.r.t weight)
        # Hint: dLoss/dWeight = 2 * x * (y_pred - y_true)
        gradient = 2 * x * (y_pred - y_true)
        total_gradient += gradient
        
    # TODO D: Calculate the average loss and average gradient across all samples
    avg_loss = total_loss / n_samples
    avg_gradient = total_gradient / n_samples
    
    # TODO E: Update the weight parameter using the average gradient and learning rate
    weight = weight - learning_rate * avg_gradient
    
    # TODO F: Print the progress metrics. 
    # Requirements: step to 2 d.p., weight, avg loss, and avg gradient to 4 d.p.
    print(f"Weight:{weight:.2f}", f"Avg loss:{avg_loss:.2f}", f"Avg gradient:{avg_gradient:.4f}")


# =====================================================================
# 4. Testing the Model After Training
# =====================================================================
print("\nTesting Model:")
test_inputs = [6.0, 7.0, 10.0, 20.0]

for x_test in test_inputs:
    # TODO G: Predict the output for the test inputs using your trained weight
    y_test_pred = weight * x_test
    print(f"Input: {x_test:.1f} -> Predicted Output: {y_test_pred:.4f}")
