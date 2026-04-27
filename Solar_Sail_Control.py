import numpy as np
import spiceypy as spice
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import itertools

C = 4.57e-6    #flux of Sun at 1 au, N/m^2
A = 150        #area of the sail, m^2 (modeled after IKAROS mission)
m = 500        #starting (post-launch) massof the spacecraft, kg (modeled after IKAROS mission)
r0 = 1.0       #reference distance from the Sun to Earth, au

#CR3BP Normalized parameters from JPL database
mu = 3.054200000000000E-6        #Sun gravitational parameter, km^3/s^2
mu_E = 398600                    #Earth gravitational parameter, km^3/s^2
DU = 149597871                   #km, DU=1 au
TU = 5022635                     #s 
m2 = mu                          #MU, mass of Earth 
m1 = 1 - mu
x2 = 1 - mu                      #normalized distance of Earth from Earth-Sun COM, au
x1 = -mu

#Halo orbit state found from JPL periodic orbits database (orbit ID 1197)
x0 = 1.0110059685997803E+0	     #au
y0 = 2.7015598908234986E-23
z0 = 4.5638537751076817E-3
vx0 = 8.8512196685031439E-16     #au/TU
vy0 = -1.1344790710813389E-2
vz0 = 1.5204310324363266E-15
state0 = np.array([x0, y0, z0, vx0, vy0, vz0])

C = 3.0007124352932899E+0        #jacobi constant (au^2/TU^2)
T = 3.0803024932763416E+0        #period (TU), = 179.065 days

def elem2state(e, a, i, w, RAAN, M0, mu):
    rp = a * (1 - e)
    state = spice.conics((rp, e, i, RAAN, w, M0, 0, mu), 0)
    return state

#Propagating the Moon using average orbital elements (assuming constant orbit)
e_M = 0.0549; a_M = 384748 / 1.496e8; i_M = 5.14 * np.pi / 180; w_M = 318.15 * np.pi / 180; RAAN_M = 125.08 * np.pi / 180; M0_M = 0     #au, rad
mu_M = 4902.8 / DU**3 * TU**2             #normalized gravitational parameter of the Moon, MU
n_M = np.sqrt(mu_E / (a_M)**3) * TU       #mean motion of the Moon, rad/TU

state0_M = elem2state(e_M, a_M, i_M, w_M, RAAN_M, M0_M, mu_E)       #state returned w.r.t. Earth
#convert x position to CR3BP rotating frame (other state components are the same because CR3B frame origin is at y=0,z=0, and Earth is considered stationary in the rotating frame)
state0_M[0] = state0_M[0] + x2

def CR3B_dyn(t, state):
    x, y, z, xdot, ydot, zdot = state
    r1 = np.sqrt((x - x1)**2 + y**2 + z**2)
    r2 = np.sqrt((x - x2)**2 + y**2 + z**2)

    rdot = np.zeros(6)
    rdot[:3] = state[3:]
    rdot[3] = (2 * ydot) + x - ((((1 - mu)/r1**3) * (x - x1))) - (((mu/r2**3) * (x - x2)))
    rdot[4] = (-2 * xdot) + y - (((1 - mu)/r1**3) * y) - ((mu/r2**3) * y)
    rdot[5] = -(1 - mu) * (z / r1**3) - (mu * z/r2**3)
    return rdot

def moon_perturber(t, rdot, state, state_M):
    rdot[3] += -mu_M * (state[0] - state_M[0]) / np.linalg.norm(state[:3] - state_M[:3])**3
    rdot[4] += -mu_M * (state[1] - state_M[1]) / np.linalg.norm(state[:3] - state_M[:3])**3
    rdot[5] += -mu_M * (state[2] - state_M[2]) / np.linalg.norm(state[:3] - state_M[:3])**3
    return rdot

def control_acceleration(t, rdot, state, alpha, delta, error_vec, m):
    x, y, z, xdot, ydot, zdot = state
    #Angular definitions from Farres & Jorba:
    #angles defining Sun-sail direction in spherical coordinates
    phi = np.arctan(y / (x + x2))
    psi = np.arctan(z / np.sqrt((x + x2)**2 + y**2))
    #angles defining sail orientation from control angles and Sun-sail direction
    nx = np.cos(phi + alpha) * np.cos(psi + delta)
    ny = np.sin(phi + alpha) * np.cos(psi + delta)
    nz = np.sin(psi + delta)

    r_Sun = state[:3] - np.array([x1, 0, 0])         #position of the spacecraft relative to the Sun in the rotating frame
    unit_r = r_Sun / np.linalg.norm(r_Sun)           #unit vector pointing from spacecraft to Sun
    unit_n = np.array([nx, ny, nz]) / np.linalg.norm(np.array([nx, ny, nz]))    #unit normal vector of the sail

    a_sail = (2*C*A/m) * (TU**2/(DU*1000)) * (r0/np.linalg.norm(r_Sun))**2 * (np.dot(unit_r, unit_n))**2 * unit_n
    #a_sail = np.array([0,0,0])                      #For testing without sail control
    rdot[3] += a_sail[0]
    rdot[4] += a_sail[1]
    rdot[5] += a_sail[2]

    #Corrective thrust control when needed
    tol = 0.001
    K_p = 16        #with sail control, gain proportion
    K_v = 10
    # K_p = 100     #without sail control
    # K_v = 10
    #ve = 2.156 * TU/DU     #exhaust velocity of thruster, km/s converted to normalized units (assuming electric propulsion)
    if np.linalg.norm(error_vec[:3]) > tol:
        #print('Error margin:', np.linalg.norm(error_vec) * 1.496e8, 'km, Applying thrust')
        acc = K_p * error_vec[:3] + K_v * error_vec[3:]
        rdot[3] += acc[0]
        rdot[4] += acc[1]
        rdot[5] += acc[2]
        #print(np.linalg.norm(acc) * DU/TU, 'km/s^2')
        delta_V = np.linalg.norm(acc) * dt
        print('Delta V used for this correction:', np.linalg.norm(acc) * dt * DU/TU, 'km/s')
        # m = m * np.exp(-delta_V / (ve))
    else:
        print('No thrust applied.')
    return rdot

def dynamics(t, state, state_M, alpha, delta, error_vec, m):
    rdot = CR3B_dyn(t, state)
    rdot = moon_perturber(t, rdot, state, state_M)
    rdot = control_acceleration(t, rdot, state, alpha, delta, error_vec, m)
    return rdot

##The following operations will need to be done at each time step for the control loop
#1. Compute the currrent state of the spacecraft using the dynamics
dt = 0.01       #time step for integration, TU
def RK4_step(function, dt, tk, state_k, state_M, alpha, delta, error_vec, m):       #General RK4 integration step
    f1 = function(tk, state_k, state_M, alpha, delta, error_vec, m)
    f2 = function(tk + dt/2, state_k + dt/2 * f1, state_M, alpha, delta, error_vec, m)
    f3 = function(tk + dt/2, state_k + dt/2 * f2, state_M, alpha, delta, error_vec, m)
    f4 = function(tk + dt, state_k + dt * f3, state_M, alpha, delta, error_vec, m)
    state_k_plus_1 = state_k + (dt/6) * (f1 + 2*f2 + 2*f3 + f4)
    return state_k_plus_1

#2. Compute the nearest point to the current state on the halo orbitstate_halo = np.array([x0, y0, z0, vx0, vy0, vz0])
halo = solve_ivp(CR3B_dyn, [0, T], state0, method='RK45', dense_output=True, rtol=1e-13, atol=1e-13)
tspan_halo = np.linspace(0, T, 1000)
halo_sol = halo.sol(tspan_halo)
r_halo = halo.sol(tspan_halo)[:3]

def nearest_point(state):
    diffs = halo_sol.T - state
    dtsts = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dtsts)
    ref_state = halo_sol[:, idx]
    #return r_halo[:, np.argmin(dists)]
    return ref_state

#3. Compute the error vector between the two
def error(state, ref_state):
    #error = nearest_point - state[:3]
    error = ref_state - state
    return error

#4. Compute the control angles to orient the solar sail in the direction of correction
def control(state, error):
    x, y, z = state[:3]
    #unit_n = error / np.linalg.norm(error)               #define unit normal vector n in the same direction as error vector
    unit_n = error[:3] / np.linalg.norm(error[:3])

    phi = np.arctan(y / (x + x2))
    psi = np.arctan(z / np.sqrt((x + x2)**2 + y**2))

    delta = np.arcsin(unit_n[2]) - psi
    alpha = np.arccos(unit_n[0] / np.cos(psi + delta)) - phi
    if abs(alpha) > np.pi/2:
        alpha = np.clip(alpha, -np.pi/2, np.pi/2)         #limit alpha and delta to be between -90 and 90 degrees (feasible solar sail)
        #print('alpha out of bounds, clipped to', alpha)
    if abs(delta) > np.pi/2:
        delta = np.clip(delta, -np.pi/2, np.pi/2)
        print('delta out of bounds, clipped to', delta)
    return alpha, delta

#5. Compute the state of the Moon at a given timestep to propagsate the next step's dynamics
def Moon_state(t):
    M_moon = M0_M + n_M * t
    state_M = elem2state(e_M, a_M, i_M, w_M, RAAN_M, M_moon, mu_E)
    state_M[0] = state_M[0] + x2
    return state_M

##Putting it all together in a control loop
#Choose an initial control state and perform first iteration(I choose solar sail facing Moon)
Moon_vec = state0_M[:3] - state0[:3]
alpha0, delta0 = control(state0, Moon_vec)
t = 0
error_vec0 = np.array([0, 0, 0, 0, 0, 0])
delta_V = 0.0

#1. Propagate the Spacecraft forward dt
state = RK4_step(dynamics, dt, t, state0, state0_M, alpha0, delta0, error_vec0, m)        #new state of spacecraft, aka state_k+1
#2. Compute nearest point on halo orbit
nearest_pt = nearest_point(state)
#3. Compute error vector between current state and nearest point
error_vec = error(state, nearest_pt)
#4. Compute control angles to orient sail in direction of error vector. Apply thrust if error is above tolerance.
alpha, delta = control(state, error_vec)
t += dt
#5. Determine the moon's current position (at time k+1) to propagate from next
state_M = Moon_state(t)                #new state of moon, aka state_k+1


#Set up the same loop for all further iterations
count = 0
states = []
while t < 10 * T:
    state = RK4_step(dynamics, dt, t, state, state_M, alpha, delta, error_vec, m)
    nearest_pt = nearest_point(state)
    error_vec = error(state, nearest_pt)
    alpha, delta = control(state, error_vec)
    t += dt
    state_M = Moon_state(t)
    tol = 0.001
    K_p = 16
    K_v = 10
    # ve = 0.2 * TU/DU
    if np.linalg.norm(error_vec[:3]) > tol:
        acc = K_p * error_vec[:3] + K_v * error_vec[3:]
        delta_V += np.linalg.norm(acc) * dt
        # m = m * np.exp(-delta_V / ve)                       #update mass based on total delta V used so far with rocket equation
        # print('mass after correction:', m, 'kg')
    count += 1
    states.append(state[:3])


#Compute total delta V used for thrust-based station keeping
print('Total delta V used for station keeping:', delta_V * DU/TU, 'km/s')

##Plotting
#Propagate Moon orbit for plotting
T_Moon = 2 * np.pi / n_M       #Moon orbital period in TU
tspan_M = np.linspace(0, T_Moon, 100)
Moon = []
for t in range(len(tspan_M)):
    state_M = Moon_state(t)
    Moon.append(state_M[:3])
Moon = np.array(Moon).T

states = np.array(states)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x2, 0, 0, color='blue', label='Earth')
ax.plot(r_halo[0], r_halo[1], r_halo[2], linestyle='--', label='Nominal Trajectory')
ax.plot(states[:,0], states[:,1], states[:,2], 'r-', label='Spacecraft Trajectory')
ax.plot(Moon[0], Moon[1], Moon[2], color='0.5', label='Moon Trajectory')
plt.legend()
plt.title('Propagation with All Controls (10 Periods)')
ax.set_xlabel('x (au)')
ax.set_ylabel('y (au)')
ax.set_zlabel('z (au)')
plt.show()