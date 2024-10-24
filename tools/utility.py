"""
Useful functions for datacubes
"""

import numpy as np


def magnitude(x, axis=None):
    "Computes the magnitude of vector x with shape (..., 3)"

    if axis==None:
        x2 = sum( x[...,i]**2 for i in range(x.shape[-1]))
    else:
        x2 = sum( x[...,i]**2 for i in axis )

    return np.sqrt(x2)
    
def gmean(x, axis=None):
    "Computes the geometric mean of |x|"

    return np.exp(np.log(np.abs(x)).mean(axis=axis))

def movingaverage(interval, window_size):
    "Computes the moving average of 'interval' over a window of 'window_size'"
    
    window = np.ones(int(window_size)) / float(window_size)
    
    return np.convolve(interval, window, 'same')
    
def turbulent(velocity):
    """
    Computes the gas turbulent velocity
    See Rea et al. (2024) for details
    """
    
    # remove any net radial outflow
    # subtract xy-average of velocity at each z

    turbvel = velocity - velocity.mean(axis=(1,2))[:,None,None,:]
    
    # remove influence of zonal flows
    # subtract y-average of y-component at each x, z
    # from y-component of vel_red

    turbvel[...,1] -= turbvel[...,1].mean(axis=1)[:,None,:]

    return turbvel

def dot(f, g):
    """
    Computes the dot product of two vectors f and g with shape (N3, N2, N1, 3)
    Uses central differencing with forward/backward differecing on the domain edges
    """
    
    if f.shape != g.shape:
        raise ValueError(f"f and g must have the same shape, but are {f.shape} and {g.shape}")
    
    fdotg = np.zeros(f.shape[:-1], f.dtype)
    
    for i in range(f.shape[-1]):
        fdotg += f[...,i]*g[...,i]
        
    return fdotg

def div(f, delta=None):
    """
    Computes the divergence of vector f with shape (N3, N2, N1, 3) in Cartesian coordinates
    Uses central differencing with forward/backward differecing on the domain edges
    """
    
    if delta is None:
        print("delta was 'None', setting delta=1 instead.")
        delta = 1
        
    divf = np.zeros(f.shape[:-1])
    
    # central difference for indices 1:-1
    divf[:,:,1:-1] += (f[:,:,2:,0] - f[:,:,:-2,0])/(2*delta)
    divf[:,1:-1,:] += (f[:,2:,:,1] - f[:,:-2,:,1])/(2*delta)
    divf[1:-1,:,:] += (f[2:,:,:,2] - f[:-2,:,:,2])/(2*delta)
    
    # forward difference for index 0
    divf[:,:,0] += (f[:,:,1,0] - f[:,:,0,0])/delta
    divf[:,0,:] += (f[:,1,:,1] - f[:,0,:,1])/delta
    divf[0,:,:] += (f[1,:,:,2] - f[0,:,:,2])/delta
    
    # backward difference for index -1
    divf[:,:,-1] += (f[:,:,-1,0] - f[:,:,-2,0])/delta
    divf[:,-1,:] += (f[:,-1,:,1] - f[:,-2,:,1])/delta
    divf[-1,:,:] += (f[-1,:,:,2] - f[-2,:,:,2])/delta
    
    return divf

def grad(f, delta=None):
    """
    Computes the gradient of scalar f in Cartesian coordinates
    Uses central differencing with forward/backward differecing on the domain edges
    """
    
    if delta is None:
        print("delta was 'None', setting delta=1 instead.")
        delta = 1
        
    gradf = np.zeros_like(f)
    
    # central difference for indices 1:-1
    gradf[:,:,1:-1,0] = (f[:,:,2:] - f[:,:,:-2])/(2*delta)
    gradf[:,1:-1,:,1] = (f[:,2:,:] - f[:,:-2,:])/(2*delta)
    gradf[1:-1,:,:,2] = (f[2:,:,:] - f[:-2,:,:])/(2*delta)
    
    # forward difference for index 0
    divf[:,:,0,0] = (f[:,:,1] - f[:,:,0])/delta
    divf[:,0,:,1] = (f[:,1,:] - f[:,0,:])/delta
    divf[0,:,:,2] = (f[1,:,:] - f[0,:,:])/delta
    
    # backward difference for index -1
    divf[:,:,-1,0] = (f[:,:,-1] - f[:,:,-2])/delta
    divf[:,-1,:,1] = (f[:,-1,:] - f[:,-2,:])/delta
    divf[-1,:,:,2] = (f[-1,:,:] - f[-2,:,:])/delta
    
    return gradf

def curl(f, delta=None):
    """
    Computes the curl of vector f with shape (N3, N2, N1, 3) in Cartesian coordinates
    Uses central differencing with forward/backward differecing on the domain edges
    """
    
    if delta is None:
        print("delta was 'None', setting delta=1 instead.")
        delta = 1
        
    curlf = np.zeros_like(f)
    
    # curlfx = dfz_dy - dfy_dz
    curlf[:,1:-1,:,0] += (f[:,2:,:,2] - f[:,:-2,:,2])/(2*delta)
    curlf[:,0,:,0]  += (f[:,1,:,2]  - f[:,0,:,2])/delta
    curlf[:,-1,:,0] += (f[:,-1,:,2] - f[:,-2,:,2])/delta
    
    curlf[1:-1,:,:,0] -= (f[2:,:,:,1] - f[:-2,:,:,1])/(2*delta)
    curlf[0,:,:,0]  -= (f[1,:,:,1]  - f[0,:,:,1])/delta
    curlf[-1,:,:,0] -= (f[-1,:,:,1] - f[-2,:,:,1])/delta
    
    # curlfy = dfx_dz - dfz_dx
    curlf[1:-1,:,:,1] += (f[2:,:,:,0] - f[:-2,:,:,0])/(2*delta)
    curlf[0,:,:,1]  += (f[1,:,:,0]  - f[0,:,:,0])/delta
    curlf[-1,:,:,1] += (f[-1,:,:,0] - f[-2,:,:,0])/delta
    
    curlf[:,:,1:-1,1] -= (f[:,:,2:,2] - f[:,:,:-2,2])/(2*delta)
    curlf[:,:,0,1]  -= (f[:,:,1,2]  - f[:,:,0,2])/delta
    curlf[:,:,-1,1] -= (f[:,:,-1,2] - f[:,:,-2,2])/delta
    
    #curlfz = dfy_dx - dfx_dy
    curlf[:,:,1:-1,2] += (f[:,:,2:,1] - f[:,:,:-2,1])/(2*delta)
    curlf[:,:,0,2]  += (f[:,:,1,1]  - f[:,:,0,1])/delta
    curlf[:,:,-1,2] += (f[:,:,-1,1] - f[:,:,-2,1])/delta
    
    curlf[:,1:-1,:,2] -= (f[:,2:,:,0] - f[:,:-2,:,0])/(2*delta)
    curlf[:,0,:,2]  -= (f[:,1,:,0]  - f[:,0,:,0])/delta
    curlf[:,-1,:,2] -= (f[:,-1,:,0] - f[:,-2,:,0])/delta
    
    return curlf

def dotdel(f, g, delta=None):
    """
    Computes the quantity (f dot del)g for vectors f and g with shape (N3, N2, N1, 3) in Cartesian coordinates
        [fx*(d/dx) + fy*(d/dy) + fz*(d/dz)] g 
    Uses central differencing with forward/backward differecing on the domain edges
    """
    
    if f.shape != g.shape:
        raise ValueError(f"f and g must have the same shape, but are {f.shape} and {g.shape}")
        
    if delta is None:
        print("delta was 'None', setting delta=1 instead.")
        delta = 1
            
    fddg = np.zeros_like(f)
    
    for i in range(3):
    
        # central difference for indices 1:-1
        fddg[:,:,1:-1,i] += f[:,:,1:-1,0] * (g[:,:,2:,i] - g[:,:,:-2,i])
        fddg[:,1:-1,:,i] += f[:,1:-1,:,1] * (g[:,2:,:,i] - g[:,:-2,:,i])
        fddg[1:-1,:,:,i] += f[1:-1,:,:,2] * (g[2:,:,:,i] - g[:-2,:,:,i])

        # forward difference for index 0
        fddg[:,:,0,i] += f[:,:,0,0] * (g[:,:,1,i] - g[:,:,0,i])
        fddg[:,0,:,i] += f[:,0,:,1] * (g[:,1,:,i] - g[:,0,:,i])
        fddg[0,:,:,i] += f[0,:,:,2] * (g[1,:,:,i] - g[0,:,:,i])

        # backward difference for index -1
        fddg[:,:,-1,i] += f[:,:,-1,0] * (g[:,:,-1,i] - g[:,:,-2,i])
        fddg[:,-1,:,i] += f[:,-1,:,1] * (g[:,-1,:,i] - g[:,-2,:,i])
        fddg[-1,:,:,i] += f[-1,:,:,2] * (g[-1,:,:,i] - g[-2,:,:,i])
    
    return fddg

def shear_map(data, deltat, x, dy, omega=1.0, order=2, reverse=False):
    "Map 'data' from y --> y - q \Omega x (t - t_n) so data is periodic in the x direction"
    
    if len(data.shape)==3: # 3D data
        axis=1
    elif len(data.shape)==2: # 2D plane in xy
        axis=-1
    else:
        print(data.shape)
        raise ValueError("Something has gone wrong in shear_map...")
    
    nx = data.shape[-1]
    
    data_shift = np.zeros_like(data, float)
        
    # compute the array of integer number of zones to shift by
    deltay = 1.5*omega*x*deltat
    shift_int = np.round(deltay/dy).astype(int)
    
    # compute the array of fractional number of zones to shift by
    shift_frac = deltay/dy - shift_int
    
    if reverse: r = -1
    else: r = 1
    
    if order==1:
        for i in range(nx):
                data_shift[...,i] = np.roll(data[...,i], r*shift_int[i], axis=1)
                
    elif order==2:
        for i in range(nx):
                data_shift[...,i] = (1.0-shift_frac[i]) * np.roll(data[...,i],
                                                                  r*shift_int[i], axis=axis) + \
                                          shift_frac[i] * np.roll(data[...,i],
                                                                  r*(shift_int[i]+1), axis=axis)
                
    else: raise ValueError("shear mapping routine must be either first or second order")

    return data_shift

def map_k(data, deltat, box_size, omega=1.0, reverse=False):
    """
    Re-map 'data' in k-space from Lagrangian wavenumber k_x = 2\pi n_x/L_x
    to Eulerian wavenumber k_x = 'k_x,L + q\Omega k_y t
    """
    
    Lx, Ly, _ = box_size
    ny = data.shape[-2]
    
    data_shift = np.zeros_like(data, complex)
        
    deltamx = 1.5*omega*(Lx/Ly)*deltat * np.arange(-ny//2, ny//2)
    
    shift_int = np.fft.fftshift(np.round(deltamx).astype(int))
    
    if reverse:
        shift_int = -shift_int
    
    if len(data.shape)==3:
        for j in range(ny):
            data_shift[:,j,:] = np.roll(data[:,j,:], shift_int[j], axis=1)
    elif len(data.shape)==2:
        for j in range(ny):
            data_shift[j,:] = np.roll(data[j,:], shift_int[j])
    else:
        raise ValueError("Something has gone wrong in map_k...")
    
    return data_shift
