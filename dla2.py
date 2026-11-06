from scipy import *
import random as r
from pylab import *
import matplotlib.animation as animation
from numba import jit

@jit(nopython=True)
def GlueSingleWalker(lattice):
    """ Generates a new random walker on a boundary, and walks it around
        until it comes close to the grown cluster. It then glues to the cluster.
    """
    N = len(lattice)
    # Random walker is released from the boundary"
    while True:
        istart = int(r.random()*4*N)
        icase = int(istart/N)    #  icase can be either 0,1,2,3
        if int(icase/2): # can be 0 or 1
            x0, y0 = istart%N, (0,N-1)[icase%2]   # either top or button of the figure, and x is random number.
        else:
            x0, y0 = (0,N-1)[icase%2], istart%N   # either left or right, and y is random number
        if lattice[x0,y0]==0: break                 # this is empty site, hence walker can start
        
    while True:  # Random walker is walking until it glues with the cloud
        x,y = x0,y0 # Random walker before being moved
        ipos = int(r.random()*4)  # one of the four moves is choosen randomly
        x_or_y = int(ipos/2)  # x_or_y is either 0 or 1
        dr = 2*int(ipos%2)-1  # dr can be -1 or +1
        if x_or_y:
            x = (x+dr)%N   # moves left one step. notice periodic boundary conditions
        else:
            y = (y+dr)%N   # moves up on step
        
        # Checking if particle can be glued!
        # Here we do not take periodic boundary conditions since they put too many particles on the boundary
        r_neighbor = x<N-1  and lattice[x+1,y] # neighbor on the right
        u_neighbor = y<N-1  and lattice[x,y+1] # neighbor above
        l_neighbor = x>0    and lattice[x-1,y] # neighbor left
        d_neighbor = y>0    and lattice[x,y-1] # neighbor below

        if r_neighbor or u_neighbor or l_neighbor or d_neighbor: # Particle must be glued
            lattice[x,y]=1
            #print 'Glued', x, y
            break  # exit the infinite loop once the walker is glued.
        x0,y0=x,y

def Mass(lattice):
    dr=20
    Rmax = sqrt(2.)/2*len(lattice) # maximum radius the particle could be
    Nr = int(Rmax/dr)+1              # number of rings
    mass = zeros(Nr)
    N=len(lattice)
    for i in range(N):
        for j in range(N):
            if lattice[i][j]==1:         # this site is occupied
                rx = i-N/2            # the distance of the point from the center in x-direction
                ry = j-N/2            # the distance of the point from the center in y-direction
                r = sqrt(rx**2+ry**2) # distance from the origin
                mass[int(r/dr)]+=1    # the ring at r/dr containe one more particle
    
    for i in range(1,Nr):
        mass[i] += mass[i-1]          # cumulative sum -> mass inside the circle of radius dr*(i+1)
    
    fi = open('mass.dat', 'w')
    for i in range(Nr):
        print(dr*(i+1), mass[i], file=fi)
    fi.close()

    
N=400
lattice = zeros((N,N),dtype=int)
lattice[int(N/2),int(N/2)]=1  # We put first point in the middle of the system

fig = figure(figsize=(8, 8)) # make figure
# make axesimage object
# the vmin and vmax here are very important to get the color map correct
im = imshow(lattice, cmap=cm.Greys, vmin=0, vmax=1, interpolation='nearest')

def updatefig(i):
    for j in range(50):
        GlueSingleWalker(lattice)
    # set the data in the axesimage object
    im.set_array(lattice)
    
    if i%50==1:
        print('Computing mass')
        Mass(lattice)
    # return the artists set
    return im,

ani = animation.FuncAnimation(fig, updatefig, frames=100, blit=True)
show()
    
