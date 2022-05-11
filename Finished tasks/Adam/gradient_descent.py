import numpy as np

# Function to compute the derivative
def derivative(x):
    return (-0.1 * ((np.exp(-x/10))* (10*np.sin(x) + np.cos(x)) ))


#Function to perform gradient descent

def gradient_descent(W, epsilon=0.01):

    #Variable to store the W value before update. This will help to check for convergence. 
    W_prev = None 
    
    # t is the iteration counter that will be used in the bias correction equations 
    t = 0

    # Save the current weights to a new list and append the updated weights in each iteration to the same
    Ws = [W]
    
    # Perform the update until convergence
    # Convergence is said to have taken place if the previous and updated weights are the same
    while (W_prev != W):
        
        # Increment the counter t for each iteration
        t += 1
        
        # Compute the gradient of W by calling the derivative function
        g = derivative(W)    
        
        # Save the W value in W_prev before update
        W_prev = W                            
        
        # Update the parameters based on the equations given in the instructions
        W = W - epsilon*g 
        
        # Append the new weight list with the udpated weight value
        Ws.append(W)
        
    return Ws, t