x = 3
y = 6

learning_rate = 0.01
weight = 0

for i in range(10):
    prediction = weight * x
    error = prediction - y
    loss = (prediction - y)**2
    gradient = 2 * x * (prediction - y)
    new_weight = weight-learning_rate * gradient


    print(f"Prediction:{prediction:4f}",f"Error:{error:4f}" ,f"Loss:{loss:4f}" ,f"New_Weight:{weight:4f}")
    weight = new_weight