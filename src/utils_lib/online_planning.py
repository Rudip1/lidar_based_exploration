#!/usr/bin/env python3
import numpy as np
import math
from time import time
import scipy.spatial
import matplotlib.pyplot as plt
import numpy as np
import random
from  utils_lib.RRT_dubins import RRT as RRT_DB # imports custom RRT* implementation

# global share map variables 
local_map = None
local_origin = None
local_resolution = None

# wraps angle into -pi to pi
def wrap_angle(angle):
    return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

class StateValidityChecker:
    """ Checks if a position or a path is valid given an occupancy map.
        #loads the map
        #checks if a postion or path are collsion free/ valid
        #tranforming between world and map coordinates
    """
    def __init__(self, distance=0.22, is_unknown_valid=True , is_rrt_star = True):
        self.map = None 
        self.resolution = None
        self.origin = None
        self.there_is_map = False
        self.distance = distance              
        self.is_unknown_valid = is_unknown_valid  
        self.is_rrt_star = is_rrt_star  


    # Initialize map details: occupancy grid, resolution, origin, and size
    def set(self, data, resolution, origin):
        global local_map, local_origin, local_resolution
        self.map = data
        self.resolution = resolution
        self.origin = np.array(origin)
        self.there_is_map = True
        self.height = data.shape[0]
        self.width = data.shape[1]
        local_map = data
        local_origin = origin
        local_resolution = resolution
    

    # Compute Euclidean distance between two 2D positions
    def get_distance( self , first_pose , second_pose):
        return math.sqrt((second_pose[0] - first_pose[0] )**2 + (second_pose[1] - first_pose[1]) **2)

# Checks if a 2D pose is valid
# checks a square area around the pose based on safety distance
    def is_valid(self, pose):

        # Convert world coordinate to discrete grid indices        
        m = self.__position_to_map__(pose)
        
        # Compute the radius in grid cells based on safety margin and map resolution
        grid_distance = int(self.distance/self.resolution)
        lower_x , lower_y = m[0] - grid_distance , m[1] - grid_distance  
        for lx in range(0,2*grid_distance):
            for ly in range(0,2*grid_distance):
                # current grid to be checked
                pose = lower_x + lx, lower_y + ly              

                # Checks if the current cell is within the map bounds
                if(self.is_on_map(pose)):   
                    # If the cell is within bounds but  with in obastcle so is invalid                        
                    if(not self.is_free(pose)): 
                        return False     
                # If the cell is out of bounds and we don't accept unknown regions consider it as invalid
                else:
                    if(not self.is_unknown_valid):
                        return False
        return True

    # Checks if a 2D pose is not valid, used for recovery
    def is_not_valid(self, pose): 
        
        m = self.__position_to_map__(pose)
        grid_distance = int(self.distance/self.resolution)
        # add 6 points upper and to lower limit 
        lower_x , lower_y = m[0] - grid_distance , m[1] - grid_distance  
        for lx in range(0,2*grid_distance):
            for ly in range(0,2*grid_distance):
                pose = lower_x + lx, lower_y + ly    
                # if(self.map[pose[0],pose[1]] >50):
                #     return pose
                # if one of the position is not free return False  , stop the loop 
                if(self.is_on_map(pose)):                           
                    if(not self.is_free(pose)): 
                        return pose     
                # if  position is not in the map bounds return  is_unknown_valid
                else:
                    if(not self.is_unknown_valid):
                        return pose
        return None

    # Validates a list of waypoints
    def check_path_smooth(self,paths):
        for path in paths:
            if(not self.is_valid(path)):
                return False
        return True
   
    # Transform position with respect the map origin to cell coordinates
    def __position_to_map__(self, p):      
        x,y = p  
        #convert x world coordnate to map index
        m_x = (x-self.origin[0])/self.resolution 
        #convert y world coordinate to map index
        m_y = (y-self.origin[1])/self.resolution
        return [round(m_x), round(m_y)]   
    def __map_to_position__(self, m):
            x ,y = m  
            # converts to world X
            w_x  = self.origin[0]+ x * self.resolution 
            # converts to world Y
            w_y  = self.origin[1] + y * self.resolution
            return [w_x, w_y]
    # returns true if the pose lies in the map bound  
    def is_on_map(self,pose):    
        if( 0<=pose[0]< self.height and 0<= pose[1] < self.width):
            return True        
        else:
            return False
    # checks if a given pose is free, unknown  
    def is_free(self, pose): 
        # if is is free return True , which means  
        if self.map[pose[0],pose[1]] == 0 :
            return True, 
        #if it is unkown return user-configured value of is_unkown_valid 
        elif self.map[pose[0],pose [1]] == -1 :
            return  self.is_unknown_valid
        return False   # cell is occupied

# Main RRT* wrapper
def compute_path(start_p, goal_p, svc, bounds = None , max_time=5.0):
    if bounds is None:
        x_min = svc.origin[0]
        x_max = svc.origin[0] + svc.width * svc.resolution
        y_min = svc.origin[1]
        y_max = svc.origin[1] + svc.height * svc.resolution
        bounds = [x_min, x_max, y_min, y_max]
    RRT = RRT_DB(svc , 3000 ,0.4, 0.3 , bounds, max_time )
    # returns the smooth path and the tree list
    path  , tree_list = RRT.compute_path(start_p, goal_p )
    # path = rrt.compute_path( start_p , goal_p)
    return path , tree_list
    
# def move_to_point(current, goal, Kv=0.5, Kw=0.5):
#     d = ((goal[0] - current[0])**2 + (goal[1] - current[1])**2)**0.5
#     psi_d = np.arctan2(goal[1] - current[1], goal[0] - current[0])
#     psi = wrap_angle(psi_d - current[2])

#     v = 0.0 if abs(psi) > 0.05 else Kv * d
#     w = Kw * psi
#     return v, w

# This is pure pursuit controller 
# follows a path by steering towards a lookahead point.
def pure_p_control(current, goal):
    # unpack current pose   
    R_x, R_y, R_yaw = current
    #handles cases where the goal may or may not include orientation:
    if len(goal) == 3:
        goal_x, goal_y, _ = goal
    else:
        goal_x, goal_y = goal

    # How far ahead the robot looks to follow
    lookahead_distance = 0.08
    # distance between the wheels- robot wheelbase
    L = 0.235 

    # vectors to goal 
    dx = goal_x - R_x
    dy = goal_y - R_y

    distance_to_goal = math.sqrt(dx**2 + dy**2)
    if distance_to_goal < lookahead_distance: 
        lookahead_x, lookahead_y = goal_x, goal_y
    else: 
        # interpolate a closer point on the way to the goal 
        scale = lookahead_distance/distance_to_goal
        lookahead_x = R_x + scale * dx
        lookahead_y = R_y + scale * dy
    #computes angle to the lookahead point
    angle_to_lookahead = math.atan2(lookahead_y - R_y, lookahead_x - R_x)
    # relative heading error between where the robot is pointing and where it should go
    alpha = wrap_angle(angle_to_lookahead - R_yaw)
    #curvature based steering angle
    delta = math.atan2(2* L * math.sin(alpha), lookahead_distance)
    # setting linear and angular velocity commands
    v = 0.4
    w = (v / L) * math.tan(delta)

    return v, w



