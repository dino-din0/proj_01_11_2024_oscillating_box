#-------------------------------------------------------
"""

    This program solves an arbitrary set of first-order
    ordinary differential equations

    All that the user would need to do is to implement the right-hand sides
    of the differential equations in the 'ode_sys' function and the
    right-hand side jacobians in the 'jacob' function, both in the
    My_Differential_System() class

    The parameters of the differential equations are 
    stored in a 'self.par[][]' array

    The remaining pieces of needed information are requested of the user at run time
"""
#-------------------------------------------------------
#-------------------------------------------------------
import numpy
import math # it may be needed for some differential equations
from   scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#-------------------------------------------------------
#-------------------------------------------------------
def main():
      
    # Class objects
    diff_sys        = My_Differential_System()   
    solve_system    = Solve_Differential_System()
      
    # object utilization
    solve_system.solve_sys( diff_sys )
      
    solve_system.print_to_file()
    #solve_system.print_to_screen()
    
    my_plot = Plot()
    
    my_plot.chart_it( solve_system.data )
    
    coordinates     = Define_points ( solve_system.comp_sol ) 
    coordinates.point_coordinates()
    
    action = Perform_animation( diff_sys.delta_t, coordinates ) 
    action.set_up_animation()


    return
#-------------------------------------------------------

#-------------------------------------------------------
class Define_Differential_System( object ):

    #---------------------------------------
    def __init__( self ):

        #self.number_of_equations_and_parameters()
        #self.init_cond_form()
        
        self.number_of_equations_and_parameters_2()
        self.init_cond_form_2()
    
        self.time_set_form()
        
        return
    #---------------------------------------
    #---------------------------------------
    def number_of_equations_and_parameters( self ):

        print()
        self.numb_of_eq     = int( input('Enter number of equations: ') )
        self.par = []
        
                
        info = str( input( 'Are parameters needed in this system? Enter yes or no: ' ) )

        if ( info == 'yes'):
            
            par_matrx = []
            for i in range( self.numb_of_eq ):
                
                print()
                print('Equation ', i + 1 )
                numb_of_par = abs( int( input('Enter number of parameters for this equation: ') ) )
                
                a = []
                for j in range( numb_of_par ):
                    
                    print('Parameter No. ', j + 1 , ' - enter this parameter: ', end = '')
                    a.append( float( input() ) )
                                
                par_matrx.append( a )
        
            a = len( par_matrx[ 0 ] )
            for i in range( 1, self.numb_of_eq ):
                if( len( par_matrx[ i ] ) > a ):
                    a = len( par_matrx[ i ] )
                        
            self.par = numpy.zeros( ( self.numb_of_eq, a ), dtype = float )  
            
            
            for i in range( self.numb_of_eq ):
                a = len( par_matrx[ i ] )
                for j in range( a ) :
                    self.par[ i ][ j ] =  par_matrx[ i ][ j ]
        
        return

    #---------------------------------------
    def init_cond_form( self ):
    
        self.q_in        =  numpy.zeros(  self.numb_of_eq, dtype = float )
       
        print()
        for i in range( self.numb_of_eq ):
            print('Initial condition No. ', i + 1, ' - enter this condition: ', end='')
            self.q_in[ i ] = float( input() )

        return

    #---------------------------------------
    #---------------------------------------
    def init_cond( self ):
     
        return( self.q_in )

    #---------------------------------------
    def time_set_form( self ):
    
        print()
        self.t_begin     =  float( input( 'Enter initial time: ' ) )
        self.t_end       =  float( input( 'Enter final time: ' ) )
        self.delta_t     =  float( input( 'Enter time step Dt: ') )
        
        
        self.t_stations = int( ( self.t_end - self.t_begin ) / self.delta_t )
        
        t_temp = self.delta_t * float( self.t_stations )        
        if ( t_temp < self.t_end ):
            self.t_end = t_temp + self.delta_t
        else :
            self.t_end = t_temp
            
        print()
        print( "Number of time stations: ", self.t_stations )
        print( "Final time: ", self.t_end )
        input()

        self.time = numpy.linspace( self.t_begin, self.t_end, self.t_stations )

        self.time_range = numpy.zeros( 2, dtype = float )
        self.time_range[ 0 ] = self.t_begin
        self.time_range[ 1 ] = self.t_end

        return

    #---------------------------------------
    def time_set( self ):
    
        return( self.time_range, self.time )
    
#-------------------------------------------------------
#-------------------------------------------------------
class My_Differential_System( Define_Differential_System ):
    
    #--------------------------------------- 
    def number_of_equations_and_parameters_2( self ):
        
        self.numb_of_eq     = 2
        self.numb_of_par     = 4 #Added this on 01/16/2024; magnitude of the force
               
        #-------------------------------------------
        # Gravitational parametermass and spring 
        
        m = 2.0
        
        k = 3.0
        
        c = 5.0 # Dampening Coefficient
        
        #Everything below here was added because of the new equation 3sin(2t)
        fs_mag = 3.0 # Sine force; Magnitude
        
        omega = 2
        
        
        #-------------------------------------------
       
        
        #-------------------------------------------   
        #numpy is a package of functions; numpy.zero initializes the matrix with just zeros
        #( self.numb_of_eq, self.numb_of_eq ) ; defines a matrix here
        #dtype ; digital type... is a floating point
        self.par = numpy.zeros( ( self.numb_of_eq, self.numb_of_par ), dtype = float ) 
              
        self.par[ 1 ][ 0 ] = k / m 
        self.par[ 1 ][ 1 ] = c / m 
        
        #Equations added from the new 3sin(2t) equation added to the sum of forces
        self.par[ 1 ][ 2 ] = fs_mag / m
        self.par[ 1 ][ 3 ] = omega
 
        return
    #---------------------------------------
    def init_cond_form_2( self ):
    
        self.q_in      =  numpy.zeros(  self.numb_of_eq, dtype = float )
               
        self.q_in[ 0 ] = 1.0
        self.q_in[ 1 ] = 3.0
            
        return

    #---------------------------------------
    def ode_sys( self, t, q ):  # implement the differential equations in this function
                                # parameters may be specified through the self.par[] array
        
        #-----------------------------------------------------
        
        dq_dt = numpy.zeros(  self.numb_of_eq, dtype = float )
        par = self.par # this will allow us to use par instead of self.par
        
        #-----------------------------------------------------
       
        dq_dt[ 0 ] = q[ 1 ]
        
        dq_dt[ 1 ] = - par[ 1 ][ 0 ] * q[ 0 ]  - par[ 1 ][ 1 ] * q[ 1 ]  + par[ 1 ][ 2 ] * math.sin( par[ 1 ][ 3 ] * t )# Modified this equation to include the effect of dampening
                
        #-----------------------------------------------------
        
        print('-> ', end = '')
        
        return( dq_dt )

    #---------------------------------------
    def jacob( self, t, q ): # implement the differential-equation jacobian in this function 
                             # parameters may be specified through the self.par[] array
        #-----------------------------------------------------
                             
        jac_mtrx = numpy.zeros( ( self.numb_of_eq, self.numb_of_eq ), dtype = float )
        par = self.par

        #-----------------------------------------------------
      
        #dq_dt[ 0 ] = q[ 1 ]
        
        jac_mtrx[ 0 ][ 0 ] = 0.0
        jac_mtrx[ 0 ][ 1 ] = 1.0   

        #dq_dt[ 1 ] = - par[ 1 ][ 0 ] * q[ 0 ]  - par[ 1 ][ 1 ] * q[ 1 ]
        #jac_mtrx = Jacobian Matrix; This ensures the accuracy of the computation
        #Jacobians are a matrix of partial derivatives containing x and v_x. The time dependent force does not affect the Jacobian.
        
        jac_mtrx[ 1 ][ 0 ] =  - par[ 1 ][ 0 ] # Replaced 0.0 with par to include the effect of dampening
        jac_mtrx[ 1 ][ 1 ] =  - par[ 1 ][ 1 ] 
       
        #-----------------------------------------------------
        
        return( jac_mtrx )

    #---------------------------------------

#-------------------------------------------------------
#-------------------------------------------------------
class Solve_Differential_System( object ):

    #---------------------------------------
    def __init__( self ):
        
        self.comp_sol = []
         
        return
    #---------------------------------------
    def solve_sys( self, diff_sys ):

        time_range, time = diff_sys.time_set()
        
        q_in = diff_sys.init_cond()
 
        print('...working...')
        self.comp_sol = solve_ivp( diff_sys.ode_sys, time_range, q_in, method = 'Radau', t_eval = time, dense_output = True, atol = 1.0e-13, rtol = 1.0e-13, jac = diff_sys.jacob )
        
        self.data_matrix()
       
        return
    
    #---------------------------------------
    def data_matrix( self ):    

        self.data = numpy.zeros( ( len( self.comp_sol.t) , len( self.comp_sol.y ) + 1 ), dtype = float )
        
        number_of_rows = len( self.data )
        
        for i in range( number_of_rows ):
            self.data[ i ][ 0 ] = self.comp_sol.t[ i ]
            j = 1
            for var in ( self.comp_sol.y ):
                self.data[ i ][ j ] = var[ i ]
                j = j + 1
                
        return

    #---------------------------------------
    def print_to_file( self ):
        
        x = self.comp_sol
        
        fs = open( 'solution.txt', 'wt')
        
        number_of_rows      = len( x.t )

        for i in range( number_of_rows ):
            string = str( x.t[ i ] ) + ' '  
            for y in ( x.y ):
                string = string + str( y[ i ] ) + ' '  

            fs.write( string )
      
            fs.write( '\n' )

        fs.close()

        return
    
    #---------------------------------------
    def print_to_screen( self ):

        x = self.comp_sol
   
        number_of_rows      = len( x.t )
        print()
        
       
        for i in range( number_of_rows ):
            string = str( x.t[ i ] ) + ' '  
            for y in ( x.y ):
                string = string + str( y[ i ] ) + ' '  
	
            print( string )

        return
#-------------------------------------------------------
#-------------------------------------------------------- 
class Plot( object ):
       
    #----------------------------------------------------
    def __init__( self ):
        
        import matplotlib.pyplot as plt
        
        self.plt = plt
    
        self.data = []  
        
        return
    #---------------------------------------------------- 
    #----------------------------------------------------    
    def chart_it( self, data ):
    
        self.i_plot = True 
        
        while ( self.i_plot == True ):
            
            self.chart_it_now( data )
            
            #i_plot = int( input( "Enter '1' for next plot or '0' to exit: ") )
        
        return
    #---------------------------------------------------- 
    #----------------------------------------------------    
    def chart_it_now( self, data ):

        number_of_rows    = len( data )
        number_of_columns = len( data[ 0 ] )
            
        #-------------------------------------------       
        def function_to_plot():
                
            i   = number_of_columns - 1
                
            if ( i > 1 ):
                
                i_msg_1 = "There are " + str( i ) + " functions in this data set \n"
                i_msg_2 = "Enter either '0' to exit or \n"
                i_msg_3 = "Enter the number of the function to plot, between 1 and " + str(i) + ": "
                i_msg   = i_msg_1 + i_msg_2 + i_msg_3             
                
            else:
                
                i_msg_1 = "There is one function in this data set \n"
                i_msg_2 = "Enter either '0' to exit or \n"
                i_msg_3 = "Enter '1' to plot the function: "
                i_msg   = i_msg_1 + i_msg_2 + i_msg_3
                
            print()
            self.ifunct = int( input( i_msg  ) )
            print()
                
            if ( self.ifunct == 0 ):
                self.i_plot = False
                
            return
        #-------------------------------------------
        #-------------------------------------------
        def data_preparation():
                
            for j in range( number_of_columns ):
                row = []
                
                for i in range( number_of_rows ):
                    row.append( data[ i ][ j ] )
            
                self.data.append( row )
                                
            self.number_of_functions = number_of_columns
            
            return
        #-------------------------------------------
            
        function_to_plot()
        
        if ( self.i_plot == True ):
            data_preparation()
           
            self.init_plot()
            self.plot_it()
            
        return
    
    #----------------------------------------------------
    #----------------------------------------------------
    def init_plot( self ):
        
        # https://matplotlib.org/stable/api/markers_api.html
       
        """
        ls = '-' 	solid    line style
        ls = '--' 	dashed   line style
        ls = '-.' 	dash-dot line style
        ls = ':' 	dotted   line style
        """
        
        """
        'r' 	Red 	
        'g' 	Green 	
        'b' 	Blue 	
        'c' 	Cyan 	
        'm' 	Magenta 	
        'y' 	Yellow 	
        'k' 	Black 	
        'w' 	White
        """
        
        self.fig, self.ax = self.plt.subplots()
        
        self.ax.grid( True )
                
        self.ax.plot( self.data[ 0 ], self.data[ self.ifunct ],  ls = "-", linewidth = 1, color = 'g', marker = 'o', ms = 4, mec = 'b', mfc = 'b'  )
        
        return
    #----------------------------------------------------         
    #----------------------------------------------------  
    def plot_it( self ):
        
        self.plt.show()
        
        return
    #---------------------------------------------------- 
       
#-------------------------------------------------------
#-------------------------------------------------------
class Define_points:
    
    #---------------------------------------
    def __init__( self, sol ):
                
        self.number_of_moving_points  = 1
               
        self.sol = sol
        
        self.number_of_rows    = len( self.sol.t )
        self.number_of_columns = len( self.sol.y )
        
        self.x = numpy.zeros( ( self.number_of_moving_points + 1, self.number_of_rows ), dtype = float )
        self.y = numpy.zeros( ( self.number_of_moving_points + 1, self.number_of_rows ), dtype = float )  
        
          
        return
    #---------------------------------------
    #---------------------------------------
    def point_coordinates( self ):
        
        x = self.x
        y = self.y
           
        #--------------------------------------- 
                 
        for i in range( self.number_of_rows ):
            #coordinates of moving points
            
            x[ 1 ][ i ] = self.sol.y[ 0 ][ i ]
            y[ 1 ][ i ] = 0.0
            
        #--------------------------------------- 
        
        self.x = x
        self.y = y
       
        return
    #---------------------------------------
    
#-------------------------------------------------------
#------------------------------------------------------------------------------    
class Perform_animation:
    
    #------------------------------------    
    def __init__( self, delta_t, coord ):
        

        self.plt = plt
        
        self.number_of_rows = coord.number_of_rows
        
        self.interval    = 0.01
        #self.history_len = int( self.number_of_rows / 5 ) # how many trajectory points to keep
        
        self.fg_x = 5
               
        self.t_a = 0.05
        self.t_b = 0.95  

        self.dt =  delta_t
        
        self.x = coord.x
        self.y = coord.y
        
        return
    #------------------------------------    
    #------------------------------------    
    def set_up_animation( self ):        
    
        
        self.fig, self.ax = self.plt.subplots()
        
        ax = self.ax

        ax.grid( True )
        
        self.line_1,  = ax.plot( [], [], ls = "-", linewidth = 0, color = 'g', marker = 's', ms = 40, mec = 'b', mfc = 'b' )

        self.time_template = 'time = %.1fs' 
        self.time_text = ax.text( self.t_a, self.t_b, '', transform = ax.transAxes )
        
        self.ax = ax 
        
        #self.ax.set_xlim( [ min( self.x[ 1 ] ), max( self.x[ 1 ] ) ] )  
        #self.ax.set_ylim( [ min( self.y[ 1 ] ), max( self.y[ 1 ] ) ] )      
              
        #self.ax.plot( [ 0.0 ], [0.0], ls = "-", linewidth = 1, color = 'y', marker = 'o', ms = 20, mec = 'r', mfc = 'y' )
        self.ax.plot( self.x[ 1 ], self.y[ 1 ], ls = "-", linewidth = 1, color = 'r', marker = ',', ms = 0, mec = 'r', mfc = 'r' )

        self.animate()
 
        return
    #------------------------------------    
    #------------------------------------    
    def animate( self ):
        
        
        ani = animation.FuncAnimation( self.fig, self.show, self.number_of_rows, interval = self.interval * self.dt, blit = True )
        plt.show()  
        
        print( ani )
    
        return
    #------------------------------------    
    #------------------------------------    
    def show( self, j ):
        
        i = 1 * j
        if ( i > self.number_of_rows ):
            i = self.number_of_rows - 1

        line_point_x_1 = [ self.x[ 1 ][ i ] ]
        line_point_y_1 = [ self.y[ 1 ][ i ] ]
        
        self.line_1.set_data( line_point_x_1, line_point_y_1 )
              
        self.time_text.set_text( self.time_template % ( i * self.dt ) )
    
        
        var = []
        
        var.append( self.line_1 )
        
        var.append( self.time_text )
        
        return( var )
    
    #------------------------------------ 
#------------------------------------------------------------------------------ 
#-------------------------------------------------------
main()
#-------------------------------------------------------





