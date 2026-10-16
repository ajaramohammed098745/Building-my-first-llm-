import torch

x = torch.tensor(4.0)
y = torch.tensor(8.0)

w = torch.tensor(
    0.5,
    requires_grad=True
)

learning_rate = 0.01

for step in range(10):

    # -------------------------
    # 1. FORWARD PASS
    # -------------------------

    prediction = w * x

    loss = (prediction - y) ** 2


    # -------------------------
    # 2. BACKWARD PASS
    # -------------------------

    loss.backward()


    print(
        f"Step {step:2d} | "
        f"Weight: {w.item():.4f} | "
        f"Prediction: {prediction.item():.4f} | "
        f"Loss: {loss.item():.4f} | "
        f"Gradient: {w.grad.item():.4f}"
    )


    # -------------------------
    # 3. UPDATE PARAMETER
    # -------------------------

    with torch.no_grad():

        # TODO:
        # Perform the gradient descent update.
        # w -=


    # -------------------------
    # 4. RESET GRADIENT
    # -------------------------

    w.grad.zero_()