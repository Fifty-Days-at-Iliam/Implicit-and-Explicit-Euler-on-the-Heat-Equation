import numpy as np
import matplotlib.pyplot as plt


#Define parameters for heat equation: u_t = k u_{x^2}:
k = 0.04
x = np.linspace(0,1,101)
dx = x[1] - x[0]

Ns = [50, 100, 700, 800, 1000, 2000, 4000, 10000, 50000]

#Define initial condition for u(x,0)
u_0= 5 + 4*np.cos(3*np.pi*x)


#Discretize function
def laplacian(u):
  N = len(u)
  lap_u = np.zeros(N)

  # Interior points
  lap_u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2

  # Boundary points (Neumann: du/dx = 0 => u_ghost = u_real_neighbor)
  # At x=0: u_{-1} = u_1. Laplacian at u_0: (u_1 - 2*u_0 + u_1) / dx^2 = 2*(u[1] - u[0]) / dx**2
  lap_u[0] = 2 * (u[1] - u[0]) / dx**2

  # At x=L: u_N = u_{N-2}. Laplacian at u_{N-1}: (u_{N-2} - 2*u_{N-1} + u_{N-2}) / dx**2 = 2*(u[N-2] - u[N-1]) / dx**2
  lap_u[N-1] = 2 * (u[N-2] - u[N-1]) / dx**2

  return lap_u

#Forward euler solver:
def forward_euler(u_0, t, dt, k):
    u=np.zeros((np.size(t),np.size(u_0)))
    u[0,:]=u_0

    for i in range(len(t)-1):
        u[i+1, :] = u[i,:]+dt*k*laplacian(u[i,:])


    return u

#Backward Euler solver:
def backward_euler(u_0,t,dt,k):
    N = np.size(u_0)
    u = np.zeros((np.size(t),N))
    u[0,:] = u_0

    #Build A = I -k dt L for internal points
    main_diag_val = (1 + 2*k*dt/dx**2)
    off_diag_val  = (-k*dt/dx**2)

    A = np.diag(main_diag_val * np.ones(N)) + \
        np.diag(off_diag_val * np.ones(N-1), 1) + \
        np.diag(off_diag_val * np.ones(N-1), -1)

    # Neumann boundary conditions (du/dx = 0)
    A[0,0] = main_diag_val # Diagonal element for u_0
    A[0,1] = off_diag_val * 2 # Coefficient for u_1 (due to ghost node u_{-1}=u_1)

    A[N-1, N-1] = main_diag_val # Diagonal element for u_{N-1}
    A[N-1, N-2] = off_diag_val * 2 # Coefficient for u_N=u_{N-2})

    for i in range(np.size(t)-1):
        u[i+1,:] = np.linalg.solve(A,u[i,:])

    return u


#Initialize arrays for simualtions and error analysis
u_forward = []
u_backward = []
u_exact = []
ts = []
rs =[]

forward_error_l2 = []
backward_error_l2 = []

forward_error_linf = []
backward_error_linf = []

#Run simulations and produce exact solutions
for i in range(len(Ns)):
  t = np.linspace(0,1,Ns[i]+1)
  dt = t[1]-t[0]
  rs.append(k*dt/dx**2)
  ts.append(t)

  u_forward.append(forward_euler(u_0, t, dt, k))
  u_backward.append(backward_euler(u_0,t, dt, k))
  u_exact.append(5 + np.outer(np.exp(-k*(3*np.pi)**2*t),4*np.cos(3*np.pi*x)))

  # Calculate L2 and L_inf errors at the final time step (t=1)
  forward_error_l2.append(np.linalg.norm(u_forward[-1][-1,:]-u_exact[-1][-1,:]))
  backward_error_l2.append(np.linalg.norm(u_backward[-1][-1,:]-u_exact[-1][-1,:]))
  

  forward_error_linf.append(np.max(np.abs(u_forward[-1][-1,:]-u_exact[-1][-1,:])))
  backward_error_linf.append(np.max(np.abs(u_backward[-1][-1,:]-u_exact[-1][-1,:])))



#Plot results:

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.loglog(rs, forward_error_l2, label='Forward')
plt.axvline(0.5, color='r', linestyle='--', label='Stability Limit (r=0.5)') # Added vertical line
plt.title('Forward L2 Error vs r Step Size')
plt.xlabel("r Step Size")
plt.ylabel("L2 Error")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.loglog(rs, backward_error_l2, label='Backward')
plt.title('Backward L2 Error vs r Step Size')
plt.xlabel("r Step Size")
plt.ylabel("L2 Error")
plt.legend()
plt.grid(True)

plt.tight_layout()


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.loglog(rs, forward_error_linf, label='Forward')
plt.axvline(0.5, color='r', linestyle='--', label='Stability Limit (r=0.5)') # Added vertical line
plt.title('Forward L_inf Error vs r Step Size')
plt.xlabel("r Step Size")
plt.ylabel("L_inf Error")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.loglog(rs, backward_error_linf, label='Backward')
plt.title('Backward L_inf Error vs r Step Size')
plt.xlabel("r Step Size")
plt.ylabel("L_inf Error")
plt.legend()
plt.grid(True)

plt.tight_layout()



# Create the figure and axes first
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for i in range(len(Ns)):
    u_f = u_forward[i]
    norm = np.linalg.norm(u_f, axis=1)
    
    if rs[i]>0.5:
        ax1.plot(ts[i], norm, '--', label=f"r={rs[i]:.2f}")


    else:
        ax2.plot(ts[i], norm, '-', label=f"r={rs[i]:.2f}")


ax1.set_title("Unstable Divergence")
ax1.set_xlabel("t")
ax1.set_ylabel(r"$||u||_2$")
ax1.grid(True)
ax1.legend()


ax2.set_title("Stable Energy Decay") 
ax2.set_xlabel("t")
ax2.set_ylabel(r"$||u||_2$")
ax2.grid(True)
ax2.legend()

plt.tight_layout()



plt.figure(figsize=(6,5))
plt.loglog(rs, forward_error_l2, 'o-', label='Forward Euler')
plt.loglog(rs, backward_error_l2, 's-', label='Backward Euler')

plt.axvline(0.5, linestyle='--',color='r', label='Stability limit')

plt.xlabel(r"$r = k \Delta t / \Delta x^2$")
plt.ylabel("$L_2$ Error at t=1")
plt.title("Stability of Forward vs Backward Euler")
plt.legend()

plt.grid(True)




stable_idx = [i for i,r in enumerate(rs) if r <= 0.5]

plt.figure(figsize=(6,5))
plt.loglog(
    [Ns[i] for i in stable_idx],
    [forward_error_l2[i] for i in stable_idx],
    'o-', label='Forward Euler'
)

plt.loglog(
    [Ns[i] for i in stable_idx],
    [backward_error_l2[i] for i in stable_idx],
    's-', label='Backward Euler'
)

plt.xlabel("Number of time steps")
plt.ylabel("$L_2$ Error")
plt.title("Convergence in Stable Regime")
plt.legend()
plt.grid(True)
plt.show()




