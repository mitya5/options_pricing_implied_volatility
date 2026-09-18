from scipy.optimize import brentq
from black_scholes import black_scholes_price, greeks

def iv_newton(market_price, s, k, t, r, option_type = "call", initial_guess = 0.2, tol = 1e-6, max_iter = 100):
    sigma = initial_guess
    for _ in range(max_iter):
        price = black_scholes_price(s,k,r, sigma,t,option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        v = greeks(s,k,r, sigma,t)["vega"]*100
        if v < 1e-10:
            raise RuntimeError("vega too small for newton_raphson")
        sigma = sigma - diff/v
        if sigma <= 0:
            raise RuntimeError("vol is non-positive, switch to brentq")
    raise RuntimeError(f"did not converge after {max_iter} iterations")

def iv_brentq(market_price, s, k, t, r, option_type = "call", sigma_bounds = (1e-6, 5.0)):
    def objective(sigma):
        return black_scholes_price(s, k, r, sigma, t, option_type) - market_price
    return brentq(objective, sigma_bounds[0], sigma_bounds[1])

def iv(market_price, s, k, t, r, option_type = "call", initial_guess = 0.2):
    try: 
        return iv_newton(market_price, s, k, t, r, option_type, initial_guess)
    except RuntimeError:
        return iv_brentq(market_price, s, k, t, r, option_type,)

if __name__ == "__main__":
    # check
    s, k, t, r = 100, 100, 1, 0.05
    true_sigma = 0.25
 
    price = black_scholes_price(s,k,r, true_sigma, t, "call")
    newton = iv_newton(price, s, k, t, r, "call")
    brent = iv_brentq(price,s, k, t, r, "call")
 
    print(f"True sigma: {true_sigma}")
    print(f"Generated call price: {price:.4f}")
    print(f"Newton: {newton:.6f}")
    print(f"Brentq{brent:.6f}")
 
    #test for itm/otm option
    print("\nstress test")
    # this will print 0 since t is so small and there is such a large dif btwn s and k
    otm_price = black_scholes_price(s, k=300, r=r, vol = .2, t=.02, option_type="call")
    print(f"otm price: {otm_price:.8f}")
    # this will just print the guess most of the time, since diff will be 0 for such an option, 
    # meaning we will fall inside of the tolerance range
    result = iv_newton(otm_price, s, k=300, t=0.02, r=r, option_type="call")
    print(f"nr result: {result:.6f}")
 

    


