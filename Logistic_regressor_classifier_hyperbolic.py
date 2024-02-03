﻿﻿import torch
import torch.nn as nn
import geoopt

"""
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, f1_score
"""
# Hyperbolic manifold definition
ball = geoopt.PoincareBall()

# Hyperbolic Linear Layer


class HyperbolicLinear(nn.Module):
    # linear operations in hyperbolic space
    def __init__(self, in_features, out_features):
        super(HyperbolicLinear, self).__init__()
        self.weight = geoopt.ManifoldParameter(
            torch.Tensor(out_features, in_features), manifold=ball)
        self.bias = geoopt.ManifoldParameter(
            torch.Tensor(out_features), manifold=ball)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization for hyperbolic space
        nn.init.uniform_(self.weight, -0.001, 0.001)
        nn.init.uniform_(self.bias, -0.001, 0.001)

    def forward(self, x):
        # Hyperbolic linear transformation|Möbius matvec
        return geoopt.mobius_matvec(self.weight, x) + self.bias

# Hyperbolic Logistic Regression Model


class HyperbolicLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(HyperbolicLogisticRegression, self).__init__()
        self.hyperbolic_linear = HyperbolicLinear(input_dim, 1)

    def forward(self, x):
        x = self.hyperbolic_linear(x)
        return torch.tanh(x)  # tanh replacing svish in Euclid ver.


# Loss function
criterion = nn.BCELoss()

# Training


def train_model(model, X_train, Y_train, num_iterations=1000, learning_rate=0.01, print_cost=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_iterations):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)
        loss.backward()
        optimizer.step()
        if print_cost and epoch % 100 == 0:
            print(
                f'Epoch [{epoch+1}/{num_iterations}], Loss: {loss.item():.4f}')
    return model

# Evaluation function w/hyperbolic distance metric


def hyperbolic_distance(x, y, c=-1):
    sqrt_c = c ** 0.5
    diff = x - y
    norm_diff = torch.norm(diff, p=2, dim=1)
    norm_x = torch.norm(x, p=2, dim=1)
    norm_y = torch.norm(y, p=2, dim=1)
    return torch.acosh(1 + 2 * (norm_diff ** 2) / ((1 - c * (norm_x ** 2)) * (1 - c * (norm_y ** 2))))


def evaluate_model(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predicted_classes = (predictions > 0.5).float()
        accuracy = (predicted_classes.eq(
            Y_test).sum() / Y_test.shape[0]).item()
        avg_hyperbolic_distance = hyperbolic_distance(
            X_test[predicted_classes == Y_test], predictions[predicted_classes == Y_test]).mean().item()
    return accuracy, avg_hyperbolic_distance


# Note: X_train, Y_train, X_test, Y_test are not ready yet
"""
model = HyperbolicLogisticRegression(input_dim=2)
trained_model = train_model(model, X_train, Y_train, print_cost=True)
accuracy, avg_hyperbolic_distance = evaluate_model(trained_model, X_test, Y_test)
print(f'Accuracy: {accuracy}, Average Hyperbolic Distance: {avg_hyperbolic_distance}')
"""
