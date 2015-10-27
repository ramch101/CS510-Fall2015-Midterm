class Attractor():
    """Attractor class to solve diffrential equations using Euler and Runge-Kutta methods.

    """
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    def __init__(self, s=10.0, p=28.0,  b=8.0/3.0): # This is the data for the class - 
        
        self.start = 0.0 # The start & end range of time component
        self.end = 80.0
        self.points = 100 # The number of increments between the point range
        self.s = s
        self.p = p
        self.b = b
        self.paramset = [self.s, self.p, self.b]
        self.solution = self.pd.DataFrame()
        self.xpoints = []
        self.ypoints = []
        self.zpoints = []
         
        self.dt = (self.end - self.start)/self.points # The incremental value dt for the functions
         

    def fx(self, x, y):
        """ 
        The defination of function fx  dx/dt = s[y(t) - x(t)] 
        
        """
        return self.s*(y - x)

    def fy(self,x,y,z):
        return ((x*(self.p-z))-y)
    
    def fz(self,x,y,z):
        """ 
        The defination of function fy  dy/dt = x(t)[p-z(t)] - y(t)

        """
        return ((x*y - self.b*z))
  
    
    def euler(self, initial_values):
        """ 
        The Euler method is used to solve the diffential equations using the formula
        y(t, x(t))

        """
        x = self.np.zeros(self.points+1)
        y = self.np.zeros(self.points+1)
        z = self.np.zeros(self.points+1) # Initialize the numpy arrays

        x[0], y[0], z[0]  = initial_values[0], initial_values[1], initial_values[2]  #   initial_values = self.paramset
    
        for i in xrange( self.points ):  # Calaculate value of the x, y and z function for a increment of dt
            x[i+1] = x[i] + self.dt*self.fx(x[i],y[i])
            y[i+1] = y[i] + self.dt*self.fy(x[i],y[i],z[i])
            z[i+1] = z[i] + self.dt*self.fz(x[i],y[i],z[i])

        return x,y, z
    
    
    
    
    def rk2(self, initial_values):
        """ 
        The Range Kutta second level method is used to solve the diffential equations using the formula 
        y(t + dt/2, x(t) + k1dt/2)
        """
        x = self.np.zeros(self.points+1)
        y = self.np.zeros(self.points+1)
        z = self.np.zeros(self.points+1) # Initialize the numpy arrays
        

        x[0], y[0], z[0]  = initial_values[0], initial_values[1], initial_values[2]  #   initial_values = self.paramset
    
        for i in xrange( self.points ):
            xk1 = self.fx( x[i],y[i])
            xk2 = self.fx(x[i]+xk1*(self.dt/2) ,y[i])
            x[i+1] = x[i] + ( xk1 + xk2 ) / 2.0
            
            yk1 = self.fy(x[i],y[i],z[i])
            yk2 = self.fy(x[i],y[i]+yk1*(self.dt/2),z[i])
            y[i+1] = y[i] + ( yk1 + yk2 ) / 2.0
            
            zk1 = self.fz(x[i],y[i],z[i])
            zk2 = self.fz(x[i],y[i],z[i]+zk1*(self.dt/2))
            z[i+1] = z[i] + ( zk1 + zk2 ) / 2.0
    
        return x,y,z   
   



    def rk4(self, initial_values):
        """ 
        The Range Kutta fourth level method is used to solve the diffential equations using the formula
        x(t + dt) = x(t) + k + 2k + 2k + k + O(dt) 6dt 

        """
        x = self.np.zeros(self.points+1)
        y = self.np.zeros(self.points+1)
        z = self.np.zeros(self.points+1) # Initialize the numpy arrays
           
        x[0], y[0], z[0]  = initial_values[0], initial_values[1], initial_values[2]  # initial_values = self.paramset
    
        for i in xrange( self.points ):
            xk1 = self.fx( x[i],y[i])
            xk2 = self.fx(x[i]+xk1*(self.dt/2) ,y[i])
            xk3 = self.fx(x[i]+xk2*(self.dt/2) ,y[i])
            xk4 = self.fx(x[i]+xk3*self.dt ,y[i])
            x[i+1] = x[i] + (self.dt/6)*( xk1 + 2*xk2 + 2*xk3 + xk4)
            
            yk1 = self.fy(x[i],y[i],z[i])
            yk2 = self.fy(x[i],y[i]+yk1*(self.dt/2),z[i])
            yk3 = self.fy(x[i],y[i]+yk2*(self.dt/2),z[i])
            yk4 = self.fy(x[i],y[i]+yk1*self.dt,z[i])
            y[i+1] = y[i] + (self.dt/6)*( yk1 + 2*yk2 + 2*yk3 + yk4)
           
            
            zk1 = self.fz(x[i],y[i],z[i])
            zk2 = self.fz(x[i],y[i],z[i]+zk1*(self.dt/2))
            zk3 = self.fz(x[i],y[i],z[i]+zk2*(self.dt/2))
            zk4 = self.fz(x[i],y[i],z[i]+zk3*self.dt)
            z[i+1] = z[i] + (self.dt/6)*( zk1 + 2*zk2 + 2*zk3 + zk4)
    
        return x,y,z   
    
             
    def evolve(self,r0=[10.0,15.0,20.0],order=4) :
        """ 
        The Evolve method is a wrapper function to initialize the parameters and call the appropriate method
        to solve the deifferential equation using the order parameter
        1 - Euler method
        2 - Runge Kutta method level 2
        4 - Runge Kutta method level 4 ( default)
        """
        x0 = r0[0] # Initialize the parameters
        y0 = r0[1]
        z0 = r0[2]
        
        if order == 1 : # call apprpriate method based on the order parameter
            self.xpoints,self.ypoints,self.zpoints = self.euler([x0,y0,z0])
        elif order == 2 :
            self.xpoints,self.ypoints,self.zpoints = self.rk2([x0,y0,z0])
        else :
            self.xpoints,self.ypoints,self.zpoints = self.rk4([x0,y0,z0])
        
        df1 = self.pd.DataFrame(self.xpoints) # collect the numpy array output and convert to a data frame for the purpose of saving to a file
        df2 = self.pd.DataFrame(self.xpoints)
        df3 = self.pd.DataFrame(self.xpoints)
        df0 = self.pd.DataFrame(self.np.linspace( self.start, self.end, self.points+1 )) # get the time component added to the data frame
        self.solution = self.pd.concat([df0, df1, df2, df3], axis=1, keys=['t','x','y', 'z']) # concat the arrays to data frames and add the column headers
        
        return self.solution
        
        
        
    def save(self) :
        """ 
        The save method is used to save the panda data frame to a file data.csv in the current working directory

        """
        self.solution.to_csv('data.csv')
        return(True) # the True status is returned from this methos to validate if the program was able to save the file sucessfully
    
        """ 
        The plot methods are used to plot the graphs of various equations 

        """
        
    def plotx(self) :
        fig = self.plt.figure()
        ax = fig.gca()
        ax.plot(self.solution.t, self.solution.x)
        ax.set_xlabel("Time")
        ax.set_ylabel("x(t)")
        ax.set_title("Solving X function")
        self.plt.show() 
    
    def ploty(self) :
        fig = self.plt.figure()
        ax = fig.gca()
        ax.plot(self.solution.t, self.solution.y)
        ax.set_xlabel("Time")
        ax.set_ylabel("y(t)")
        ax.set_title("Solving Y function")
        self.plt.show() 
 
    def plotz(self) :
        fig = self.plt.figure()
        ax = fig.gca()
        ax.plot(self.solution.t, self.solution.z)
        ax.set_xlabel("Time")
        ax.set_ylabel("z(t)")
        ax.set_title("Solving Z function")
        self.plt.show()  

    def plotxy(self) :
        fig = self.plt.figure()
        ax = fig.gca()
        ax.plot(self.solution.x, self.solution.y)
        ax.set_xlabel("x(t)")
        ax.set_ylabel("y(t)")
        ax.set_title("Display x(t) vs y(t) ")
        self.plt.show()       
        
    def plotyz(self) :
        fig = self.plt.figure()
        ax = fig.gca()
        ax.plot(self.solution.y, self.solution.z)
        ax.set_xlabel("y(t)")
        ax.set_ylabel("z(t)")
        ax.set_title("Display y(t) vs z(t) ")
        self.plt.show()        
    
    def plotzx(self) :
        fig = self.plt.figure()
        ax = fig.gca()
        ax.plot(self.solution.z, self.solution.x)
        ax.set_xlabel("z(t)")
        ax.set_ylabel("x(t)")
        ax.set_title("Display z(t) vs x(t) ")
        self.plt.show()      
   
    def plot3d(self) :
        fig = self.plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.xpoints, self.ypoints, self.zpoints) # in this example we pass numpy arrays vs panda data frame, another way to plot the graphs
        ax.set_xlabel("x(t)")
        ax.set_ylabel("y(t)")
        ax.set_zlabel("z(t)")
        ax.set_title("Solution Curves of x, y, z equations ")
        self.plt.show()          