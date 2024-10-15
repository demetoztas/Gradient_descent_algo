import numpy as np
import pandas as pd


data = pd.read_csv("house_price.csv")
X = data[['area', 'rooms']].values 
y = data['price'].values  


X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  
y = (y - np.mean(y)) / np.std(y)  


X = np.hstack([np.ones((X.shape[0], 1)), X])


def compute_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)


def gradient_descent(X, y, theta, learning_rate, iterations, tolerance):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)
        theta -= (learning_rate / m) * (X.T.dot(predictions - y))
        cost_history[i] = compute_cost(X, y, theta)

        
        if (i + 1) % 20 == 0:  
            print(f"İterasyon {i + 1}: Parametreler: {theta}, Hata: {cost_history[i]}")

        
        if i > 0 and abs(cost_history[i] - cost_history[i - 1]) < tolerance:
            print(f"Hata, {tolerance} eşiğinin altına düştü. Eğitim sonlandırılıyor.")
            break

    return theta, cost_history



theta = np.zeros(X.shape[1])  
learning_rate = 0.0009
iterations = 1000
tolerance = 1e-5 

theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations, tolerance)


final_cost = cost_history[min(len(cost_history) - 1, len(cost_history) - 1)]
print("Eğitilmiş Parametreler:", theta)
print("Son Hata Değeri:", final_cost)

