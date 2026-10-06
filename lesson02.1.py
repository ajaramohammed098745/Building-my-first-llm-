training_data = [
    (1, 3), (4, 9), (3, 6), (5, 9)
]

weight = 0.0
learning_rate = 0.1

for i in range (30):
  total_loss = 0.00
  total_gradient = 0.00
  num_samples = len(training_data)

  for x, y in training_data:
    y_results = weight * x

    loss = (y_results - y)**2
    total_loss += loss

    gradient = 2 * x *(y_results- y)
    total_gradient += gradient
    
  avg_loss= total_loss / num_samples
  avg_gradient = gradient / num_samples

  weight = weight - learning_rate * avg_gradient

  print(f"weight: {weight:.2f}-> prediction: {y_results:.4f}-> loss:{loss:.4f}")


print('\ntraining model')
test_values = [
    1, 9 ,5 , 27, 4, 19
  ]

for x_true in test_values:
    y_pred = weight * x_true
    print(f"Input: {x_true:.1f} -> Predicted Output: {y_pred:.4f}")