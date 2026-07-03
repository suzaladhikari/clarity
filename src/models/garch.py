import numpy as np 
class Garch: 
    def __init__(self, omega, alpha, beta):
        self.omega = omega ## Even if nothing happened yesterday, volatility never goes ot zero. This isthe floor
        self.alpha = alpha ## How much we care about yesterday's shock. High alpha = market is jumpy, reacts hard to surprises. 
        self.beta = beta ## How much yesterday's volatility carries into today. High beta == volatility is persistent, slow to die down 

    def computing_variance(self, returns):
        n = len(returns) ## This is the return of the n number of days. 
        variance = np.zeros(n) ## Setting up the variance to be zero  as the size of the returns
        variance[0] = np.var(returns) ## The first variance is the variance of the total returns
        for t in range(1,n): ## THen we keep on updating the variance 
            variance[t] = self.omega + self.alpha * returns[t-1] **2 + self.beta * variance[t-1]
        return variance  ## It returns the list of variance where the index represents each day varaince, 
    
    def forecast_next_day(self, returns, variance):
        last_return = returns[-1]
        last_variance = variance[-1]
        next_day_variance = self.omega + self.alpha * last_return **2 + self.beta * last_variance
        return next_day_variance
    