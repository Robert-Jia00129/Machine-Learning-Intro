import numpy as np

# Function to compute the response data given the predictor data
def get_response_data(x):
    return np.cos(x) * np.exp(-x/10)

# Function to compute the derivative
def derivative(x):
    return (-0.1 * ((np.exp(-x/10))* (10*np.sin(x) + np.cos(x)) ))


def lr_decay(W, epsilon,decay_rate = 0.1,delta=1e-8):

    #Variable to store the W value before update. This will help to check for convergence. 
    W_prev = None
    t = 0
    
    Ws = [W]
    lrs = [epsilon]
    
    # Perform the update until convergence
    # Convergence is said to have taken place if the previous and updated weights are the same
    while (W_prev != W):
        
        # Increment the counter t for each iteration
        t += 1
        
        # Compute the gradient by calling the derivative function
        g = derivative(W)  
         
        
        # Save the W value in W_prev before update
        W_prev = W   
        
        epsilon = (1-decay_rate)*epsilon
        
        # Update the parameters based on the equations given in the instructions
        W = W - epsilon*g
        
        # Append the new weight list with the udpated weight value
        Ws.append(W)
        lrs.append(epsilon)
        
    return Ws,lrs

def rms_prop(W, epsilon, rho2=0.999, delta=1e-8):

    #Variable to store the W value before update. This will help to check for convergence. 
    W_prev = 0
    
    #Inititalise v and r to zero
    r = 0 
    
    # t is the iteration counter that will be used in the bias correction equations 
    t = 0

    # Save the current weights to a new list and append the updated weights in each iteration to the same
    Ws = [W]
    lrs = [epsilon]
    
    # Perform the update until convergence
    # Convergence is said to have taken place if the previous and updated weights are the same
    while (np.abs(W_prev -W) > delta):
        
        # Increment the counter t for each iteration
        t += 1
        
        # Compute the gradient by calling the derivative function
        g = derivative(W)  
         
        
        # Update the r, the moving average of the  sqaured gradient according to the equation given in the instructions
        r = rho2*r + (1-rho2)*(g**2)
        
        # According the the bias correct equations get the corrected v and r values     
        r_bias_corr = r/(1-(rho2**t))        
        
        # Save the W value in W_prev before update
        W_prev = W                            
        
        # Update the parameters based on the equations given in the instructions
        W = W - (epsilon*g/(np.sqrt(r_bias_corr)+delta))   
        
        # Append the new weight list with the udpated weight value
        Ws.append(W)
        lrs.append((epsilon/(np.sqrt(r_bias_corr)+delta)))
        
    return Ws,lrs