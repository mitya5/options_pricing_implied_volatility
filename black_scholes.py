from scipy.stats import norm
import numpy as np

def calculate_deltas(s,k,vol,t):
    d_1 = (np.log(s/k) + (.5 * vol**2)*t)/(vol * np.sqrt(t))
    d_2 = d_1 - (vol * np.sqrt(t))
    return (d_1, d_2)

def black_scholes_price(s, k, r, vol, t, option_type="call"):
    d_1, d_2 = calculate_deltas(s,k,vol,t)
    if option_type == "call":
        price = s * norm.cdf(d_1) - k * np.exp(-r*t) * norm.cdf(d_2)
    elif option_type == "put":
        price = k * np.exp(-r*t) * norm.cdf(-d_2) - s * norm.cdf(-d_1)
    else:
        raise ValueError("invalid input. input either 'call' or 'put'")
    return price

def greeks(s, k, r, vol, t, option_type = "call"):
    # vega is per 1% change in vol, theta is in calendar-day convention
    d_1, d_2 = calculate_deltas(s,k,vol,t)
    if option_type == "call":
        delta = norm.cdf(d_1)
        theta = (- (s * norm.pdf(d_1) * vol) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d_2))/252
        rho = k * t * np.exp(-r * t) * norm.cdf(d_2)
    elif option_type == "put":
        delta = norm.cdf(d_1) - 1
        theta = (- (s * norm.pdf(d_1) * vol) / (2 * np.sqrt(t)) + r * k * np.exp(-r * t) * norm.cdf(-d_2))/252
        rho = -k * t * np.exp(-r * t) * norm.cdf(-d_2)
    else:
        raise ValueError("invalid input. input either 'call' or 'put'")
    gamma = norm.pdf(d_1) / (s*vol*np.sqrt(t))
    vega = s * norm.pdf(d_1) * np.sqrt(t) / 100 
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


if __name__ == "__main__":
    # check
    s, k, r, vol, t = 100, 100, .05, .2, 1
    print(f"put: {black_scholes_price(s,k,r,vol,t)}")
    print(f"call: {black_scholes_price(s,k,r,vol,t, option_type='put')}")
    print(greeks(s, k, r, vol, t))
    print(greeks(s,k,r,vol,t,option_type = "put"))

