import matplotlib.pyplot as plt
import numpy as np

FUNC_RANGE = (0.1, 3) # the part of the function we will focus on
x = np.linspace(*FUNC_RANGE, 200)

def f(x):
    return np.cos(3*np.pi*x)/x

def der_f(x):
    '''derivative of f(x)'''
    return -(3*np.pi*x*np.sin(3*np.pi*x)+np.cos(3*np.pi*x))/x**2

def get_tangent_line(x, x_range=.5):
    y = f(x)
    m = der_f(x)
    # get tangent line points
    # slope point form: y-y_1 = m(x-x_1)
    # y = m(x-x_1)+y_1
    x1, y1 = x, y
    x = np.linspace(x1-x_range/2, x1+x_range/2, 50)
    y = m*(x-x1)+y1
    return x, y, m

def plot_it(cur_x, title='', ax=plt):
    y = f(x)
    ax.plot(x,y)
    ax.scatter(cur_x, f(cur_x), c='r', s=80, alpha=1);
    x_tan, y_tan, der = get_tangent_line(cur_x)
    ax.plot(x_tan, y_tan, ls='--', c='r')
    # indicate when if our location is outside the x range
    if cur_x > x.max():
        ax.axvline(x.max(), c='r', lw=3)
        ax.arrow(x.max()/1.6, y.max()/2, x.max()/5, 0, color='r', head_width=.25)
    if cur_x < x.min():
        ax.axvline(x.min(), c='r', lw=3)
        ax.arrow(x.max()/2.5, y.max()/2, -x.max()/5, 0, color='r', head_width=.25)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-3.5, 3.5)
    ax.set_title(title)
    
def clip(g, clip_threshold=8):
    '''return clipped gradient with a magnitude <= clip_threshold'''
    # your code here
    if np.abs(g) > clip_threshold:
        g = g*clip_threshold/np.abs(g)
    # end of your code
    return g