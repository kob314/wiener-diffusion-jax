"""Wiener first-passage computations extracted from the example notebook."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

def jax_wfpt_aux(v,w,a,t,tt,use_small_t,k):
    """
    Evaluate the lower-boundary Wiener first-passage-time (wfpt) density.
    Auxiliary function used by jax_wfpt.

    Parameters
    ----------
    v : drift rate.
    w : relative starting position between the boundaries (0 < w < 1).
    a : boundary separation (a > 0); the diffusion scale is fixed to one.
    t : positive decision-time grid, shaped (n_times, 1).
    tt : rescaled time, t / a**2.
    use_small_t : Boolean mask selecting the small-time series.
    k : number of retained terms in each series; must be odd.

    Returns
    -------
    Density values with shape (n_times, 1).
    """

    pi = jnp.pi

    # Series indices: ks for the small-time expansion, kl for the
    # large-time expansion. For k=21, these are -10,...,10 and 1,...,21.
    # Each has shape (1, k), broadcasting against the time column.
    ks = jnp.arange(-(k-1)/2,(k-1)/2+1,1,dtype='int32').reshape(1,-1)      # +1 is due to numpy arange convention
    kl = jnp.arange(1,k+1,1,dtype='int32').reshape(1,-1)                   # +1 is due to numpy arange convention

    # ps and pl are the small-time and large-time approximations to
    # the zero-drift density on the rescaled time axis.
    # Sum over series terms, retaining the time column.
    ps = jnp.sum((w+2*ks)*jnp.exp(-((w+2*ks)**2)/(2*tt)), axis=1, keepdims=1) / jnp.sqrt(2*pi* tt**3)
    pl = jnp.sum(kl*jnp.exp(-(kl**2 * pi**2)*tt/2) * jnp.sin(kl*pi*w), axis=1, keepdims=1)*pi
    
    # Select the expansion using the error-based mask from jax_wfpt.
    p  = jnp.where(use_small_t, ps, pl)

    # Apply the drift correction and the 1/a**2 time-rescaling factor.
    p  = jnp.exp( jnp.log(p) - (v*a*w + (v**2)*t/2) - 2*jnp.log(a) )

    return p



def jax_wfpt(v,w,a,grid_t,k):

    """
    Approximate decision-time bin masses for both response boundaries.

    Parameters
    ----------
    # v:  drift rate
    # w:  relative starting point
    # a:  bound separation
    # grid_t: rt grid
    # k:  number of terms in the series expansion (must be odd)

    Returns
    -------
    Array with shape (n_times, 2):
    column 0 represents the lower boundary, column 1 the upper boundary.
    Each column sums to its boundary's approximate cumulative hitting
    probability at grid_t[-1]. The combined sum can therefore be below one.
    """
    
    pi = jnp.pi

    # t is decision time; tt is time rescaled to unit boundary separation.
    t  = grid_t
    tt = t/(a**2)

    # log_eps_s and log_eps_l are log truncation-error bounds for the
    # small-time and large-time expansions, respectively. (log space for numerical stability)
    log_eps_s = -jnp.log(jnp.sqrt(8*pi*tt)) - (k-2)**2/(2*tt)       # small t expansion error upper bound
    log_eps_l = -jnp.log(pi*tt)             - tt*(k**2 * pi**2)/2   # large t expansion error upper bound
    use_small_t = jnp.where( ( (log_eps_s <= log_eps_l) & (tt < (k-1)**2) ) | (tt < 1.0/(pi**2 * k**2)) , True, False)    
    
    # RT distribution for the lower and upper bound:
    p_W = jnp.concatenate( (jax_wfpt_aux(v,w,a,t,tt,use_small_t,k),  jax_wfpt_aux(-v,1-w,a,t,tt,use_small_t,k) ),axis=1)
    p_W = p_W/jnp.sum(p_W,axis=0,keepdims=True)
    p_W = p_W * jnp.concatenate( (jax_wfpt_aux_cum(v,w,a,t,k),  jax_wfpt_aux_cum(-v,1-w,a,t,k) ),axis=1) #  normalizing with the cumulative densities
    
    return p_W

def jax_wfpt_aux_cum(v,w,a,t,k):
    """
    Approximate the lower-boundary cumulative probability at t[-1].

    Parameters
    ----------
    # v:  drift rate
    # w:  relative starting point
    # a:  bound separation
    # grid_t: rt grid
    # k:  number of terms in the series expansion (must be odd)
    
    """
    
    pi = jnp.pi
    
    # Positive integer indices for the k retained large-time series terms.
    kl = jnp.arange(1,k+1,1,dtype='int32').reshape(1,-1)                   # +1 is due to numpy arange convention

    # P is the eventual lower-boundary hitting probability, with no
    # time limit. For near-zero drift, its limiting value is 1-w.
    P = jnp.where(jnp.abs(v*a) > 10**-5, (1-jnp.exp(-2*v*a*(1-w)))/(jnp.exp(2*v*a*w)-jnp.exp(-2*v*a*(1-w))), 1-w)

    # F_aux is the series component of the probability of hitting the
    # lower boundary after t[-1]. The prefactor is applied below.
    F_aux = jnp.sum( kl*jnp.sin(pi*kl*w)/(v**2 + (kl*pi/a)**2) *jnp.exp(-t[-1]/2*(kl*pi/a)**2) , axis=1, keepdims=1)
    F = P - 2*pi/(a**2)* jnp.exp( -(v*a*w + (v**2)*t[-1]/2) ) * F_aux
    
    return F

def jax_wfpt_noisy(v,w,a,t0,t0_lsig,l,grid_t,k):

    """
    Combine Wiener decision times, variable nondecision times, and lapses.

    Parameters
    ----------
    v : drift rate
    w : relative starting position 
    a : boundary separation 
    t0 : median of the lognormal nondecision-time distribution
    t0_lsig : standard deviation of log nondecision time
    l : lapse probability (0 <= l <= 1).
    grid_t : positive, uniformly spaced time grid, shape (n_times, 1).
    k : number of retained Wiener series terms; must be odd.

    Returns
    -------
    Array with shape (n_times, 2), containing approximate reaction-time
    bin masses for lower-boundary responses (column 0) and upper-boundary
    responses (column 1). Its total mass can be below one because times
    outside the retained window are excluded.

    """

    # Wiener pdf
    p_W = jax_wfpt(v,w,a,grid_t,k)

    # non-decision time pdf
    p_t0_noise = jsp.stats.norm.pdf(jnp.log(grid_t),jnp.log(t0),t0_lsig) / grid_t
    p_t0_noise = p_t0_noise/jnp.sum(p_t0_noise) * jax.scipy.stats.norm.cdf( jnp.log(grid_t[-1]),jnp.log(t0),t0_lsig )
    
    
    # Wiener + noisy non-decision time pdf
    p_aux = jnp.concatenate( ( jnp.convolve(p_t0_noise.reshape(-1),p_W[:,0].reshape(-1)).reshape(-1,1) , jnp.convolve(p_t0_noise.reshape(-1),p_W[:,1].reshape(-1)).reshape(-1,1) ), axis=1)
    p = p_aux[0:(grid_t.size),:]
    
    p = p/jnp.maximum(jnp.sum(p),1) # ensure that the integral of probability is not bigger then 1
    
    #Wiener + noisy non-decision time + lapse pdf
    p_lapse = grid_t*0+1
    p_lapse = p_lapse/sum(p_lapse)
    p = (1.0-l)*p + l*p_lapse/2

    return p 
