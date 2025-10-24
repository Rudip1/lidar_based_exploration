#!/usr/bin/env python3


########################
# RRT_Star using using Dubins path#
#######################
import time
import random
import dubins
import math
from scipy.spatial import cKDTree
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image 
from math import sqrt

# class that represent a node in the RRT tree
class Node:
    def __init__(self , x , y  ,yaw = 0):
        self.x = x 
        self.y = y 
        self.yaw = yaw 
        self.id = 0  
        self.f_score = float('inf')
        self.g_score = float('inf') 
        self.parent  = None

    # calculate the distance between the current node and the target node
    def get_distance(self,target):
        distance = sqrt( (self.x-target.x)**2 + (self.y-target.y)**2 )
        return distance 
    # find the nearest node from the 
    # def nearest_node(self,nodes ):
    #     min_distance = float('inf')
    #     nearest_node = None
    #     for node in nodes:
    #         distance = self.get_distance(node)
    #         if(distance < min_distance and distance >  0.0001 and node != self):
    #             nearest_node = node
    #             min_distance = distance
            
    #     return nearest_node
    def nearest_node(self, nodes):
        nodes_array = np.array([(node.x, node.y) for node in nodes])
        tree = cKDTree(nodes_array)
        dist, idx = tree.query([self.x, self.y], k=1)
        return list(nodes)[idx]
    # used for optimal parent selection
    def nodes_with_in_radius(self, nodes, radius , k = 20):

        nodes = list(nodes)
        # builds a 2D NumPy array of the positions of all nodes
        nodes_array = np.array([(node.x, node.y) for node in nodes])
        # Create a KDTree from nodes for fast spatial queries
        tree = cKDTree(nodes_array)
        # Ensures we don't ask more neighbors the available nodes
        k = min(k, len(nodes))
        # gets k nearest neighbors to current node
        distances, indices = tree.query([self.x, self.y], k)
       
        if k == 1:
            # warapping into arrays
            distances = np.array([distances])
            indices = np.array([indices])

        distances = distances.flatten()
        indices = indices.flatten()
        # Filter the k nearest nodes based on the radius
        nodes_with_in_radius = [nodes[n] for i,n  in enumerate(indices) if distances[i] <= radius and nodes[n] != self]

        return nodes_with_in_radius
    # for debugging clarity
    def __str__(self):
        return str(self.id)
# class that represent the RRT tree
class RRT:
    def __init__(self,svc,k,q,p,dominion = [-10,10,-10,10] ,max_time = 7, is_RRT_star = True):
        
        self.svc = svc
        self.k = k
        self.q = q
        self.p = p 
        # self.dominion = dominion
        #self.dominion  = [-5.0,5.7,-5.0,4.8]
        map_x_min = self.svc.origin[0]
        map_x_max = self.svc.origin[0] + self.svc.width * self.svc.resolution
        map_y_min = self.svc.origin[1]
        map_y_max = self.svc.origin[1] + self.svc.height * self.svc.resolution
        self.dominion = [map_x_min, map_x_max, map_y_min, map_y_max]

        self.node_list = {}
        self.max = max_time 
        self.vertices  = []
        self.edges  = []
        self.node_counter = 1
        self.path = []
        self.smoothed_path = []
        # self.goal_index = []
        self.is_RRT_star = is_RRT_star
        self.radius = 3.0    # radius used in RRT* rewire
        self.max_time = max_time # max time for planning
        self.goal_found = False
        #dubins sampling resolution
        self.step_size = 0.1
        #turning radius for dubins path
        self.debiun_radius = 0.2
    #chooses the best parent for a new node for RRT*
    def optimal_parent(self,qnew,current_parent):
        self.vertices = self.node_list.keys()
        # filter a nodes with a given radius 
        nodes_with_in_radius = qnew.nodes_with_in_radius(self.vertices,self.radius)
        best_cost = self.node_list[current_parent] + qnew.get_distance(current_parent)
        # best_cost = current_parent.g_score + qnew.get_distance(current_parent)
        best_parent = current_parent
        path = []
        #if no candidate found, returns the default
        if not nodes_with_in_radius:
            return best_parent , []
        else :  
            # for each near by nodes
            for node in nodes_with_in_radius:
                # Get the yaw of the node
                yaw = self.wrap_angle(math.atan2(qnew.y - node.y, qnew.x - node.x))
                # collision_free = self.svc.check_path([[node.x,node.y],[qnew.x,qnew.y]])
                collision_free , dubins_path = self.dubins_check(node,qnew)
                #new_node_cost = self.node_list[node] + qnew.get_distance(node)
                new_node_cost = self.node_list[node] +len(dubins_path)
                
                if new_node_cost < best_cost and collision_free:
                    best_parent = node
                    path = dubins_path
            return best_parent , path
    #improves the tree by rewiring nearby nodes through the new one if it is cheaper.
    def rewire(self,qnew):
        # filter a nodes with a given radius 
        self.vertices = self.node_list.keys()
        nodes_with_in_radius = qnew.nodes_with_in_radius(self.vertices,self.radius)
        for node in  nodes_with_in_radius:
            # new_node_cost = self.node_list[node] + qnew.get_distance(node)
            new_node_cost = self.node_list[node] + len(qnew.debiuns_path)
            collision_free , debiuns_path = self.dubins_check(qnew,node)
            # collision_free = self.svc.check_path([[qnew.x,qnew.y],[node.x,node.y]])   
            if new_node_cost < self.node_list[node] and collision_free:
                node.parent = qnew
                self.edges.remove((node.parent,node))
                self.edges.append((qnew,node))
                self.node_list[node] = new_node_cost
                self.node.debiuns_path = debiuns_path
                # required to generate proper future dubins path
                self.node.yaw = math.atan2(qnew.y - node.y, qnew.x - node.x)
    # def rewire(self, qnew):
    #     # filter a nodes with a given radius 
    #     self.vertices = self.node_list.keys()
    #     nodes_with_in_radius = qnew.nodes_with_in_radius(self.vertices, self.radius)
    #     for node in nodes_with_in_radius:
    #         # heading angle difference check
    #         yaw_diff = abs(self.wrap_angle(qnew.yaw - node.yaw))
    #         if yaw_diff > math.radians(60):  # reject sharp heading transitions
    #             continue

    #         new_node_cost = self.node_list[node] + len(qnew.debiuns_path)
    #         collision_free, debiuns_path = self.dubins_check(qnew, node)

    #         if new_node_cost < self.node_list[node] and collision_free:
    #             node.parent = qnew
    #             self.edges.remove((node.parent, node))
    #             self.edges.append((qnew, node))
    #             self.node_list[node] = new_node_cost
    #             self.node.debiuns_path = debiuns_path
    #             self.node.yaw = math.atan2(qnew.y - node.y, qnew.x - node.x)
    #incase of rewiring used to update g_score
    #used to sample random configuration with bias towards goal
    def Rand_Config(self):       
        prob = random.random() # generate a random number between 0 and 1
        # generate a random node with in the dominion 
        x = random.uniform(self.dominion[0],self.dominion[1])
        y = random.uniform(self.dominion[2],self.dominion[3])
        x = np.clip(x, self.dominion[0], self.dominion[1])
        y = np.clip(y, self.dominion[2], self.dominion[3])
        qrand = Node(x,y)
        # if the random number is less than the probability of selecting the goal
        if(prob < self.p):          
           qrand = self.goal 
        return qrand  
    # identifies the best node to extend the tree          
    def Near_Vertices(self , qrand):
        self.vertices = self.node_list.keys()
        qnear =  qrand.nearest_node(self.vertices ) # find the nearest node from the vertices
        return qnear
    # to extend the tree from qnear to qrand
    def New_Config(self , qnear , qrand):
        dir_vector = np.array([qrand.x -  qnear.x , qrand.y - qnear.y])
        length = qnear.get_distance(qrand)
        if(length == 0):
            return qrand
        norm_vector = dir_vector/length # normalize the dir vector
        # if sampled node is closer than the step size q , don't shorten the step
        if(self.q > length):
            return qrand
        qnew = np.array([qnear.x,qnear.y]) + norm_vector*self.q      
        qnew = Node(qnew[0] , qnew[1])
        return qnew
    # returns final smooth dubins path from gaol to start
    #converts parent pointer tree to a dubins-smoothed path 
    def reconstruct_db_path(self):
        self.debiuns_path = []
        self.path = []
        node = self.goal
        while node.parent:# follow parent pointer
            for p in reversed(node.debiuns_path):
                self.debiuns_path.append(p)
            self.path.append((node.x, node.y))
            node = node.parent
        self.path.append((self.start.x,self.start.y)) #finally add start point 
        self.path.reverse()
        self.debiuns_path.reverse()
        return self.debiuns_path
     # returns a raw node-to-node path not interpolated like dubins   
    
    def wrap_angle(self,angle):
       return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    # get the tree of the RRT
    def get_tree(self):
        tree_list = [[[edge[0].x , edge[0].y] ,[edge[1].x , edge[1].y]] for edge in self.edges]
        return tree_list

    # Generates a Dubins path between two nodes
    def dubins_check(self, from_node, to_node):
        # Compute desired goal heading based on direction to target
        to_yaw = self.wrap_angle(math.atan2(to_node.y - from_node.y, to_node.x - from_node.x))

        # Create Dubins path from start (x, y, yaw) to goal (x, y, to_yaw)
        path_db = dubins.shortest_path(
            (from_node.x, from_node.y, from_node.yaw),
            (to_node.x, to_node.y, to_yaw),
            self.debiun_radius)

        # Sample the Dubins path at fixed step intervals
        waypoints = path_db.sample_many(self.step_size)

        # Collect (x, y) pairs for collision checking and reconstruction
        debiuns_path = [[ix, iy] for (ix, iy, _) in waypoints[0]]

        # Check if path is valid, now intermediate point lie inside obstale
        collision_free = self.svc.check_path_smooth(debiuns_path)
        return collision_free, debiuns_path

    # computes collision free dubins path from start to goal using RRT*
    def compute_path(self , start , goal):
        #starts timer
        self.start_time = time.time()

        # start with yaw
        self.start = Node(start[0],start[1] ,start[2])   
        # goal
        self.goal =  Node(goal[0],goal[1])
        # add start node to the RRT
        self.node_list[self.start] = 0
    
        for k in range(self.k):
            qrand = self.Rand_Config()
            while(not self.svc.is_valid([qrand.x,qrand.y])):# rejects if it is inside obstacle
               qrand = self.Rand_Config()
            qnear = self.Near_Vertices(qrand)
            qnew  = self.New_Config(qnear,qrand)
            # generates dubins connection
            collision_free , debiuns_path = self.dubins_check(qnear,qnew)
            if collision_free :    
                if(True): #allows optional parent optimization
                     # find better parent than qnear if any
                     qnear,path = self.optimal_parent(qnew,qnear)
                     if(path):
                         debiuns_path = path
                new_cost = self.node_list[qnear] + len(debiuns_path)
               # if qnew is not in the tree or if the new path is cheaper
                if qnew not in self.node_list or self.node_list[qnew] > new_cost:
                    self.node_list[qnew] = new_cost
                    qnew.parent = qnear
                    qnew.debiuns_path = debiuns_path
                    #heading direction for the next dubins path curve
                    #because next it will start from qnew
                    qnew.yaw = math.atan2(qnew.y - qnear.y, qnew.x - qnear.x)
                    self.edges.append((qnear,qnew))   
                    self.node_counter += 1     
                # to imporeve path cost 
                if( self.is_RRT_star): 
                    self.rewire(qnew)
                    pass
                if(qnew == self.goal):
                    self.goal_found = True
            if((time.time() - self.start_time) > self.max_time and not self.goal_found ):
                self.max_time += 0.5 # give additional time to search
            elif(self.goal_found and (time.time() - self.start_time) > self.max_time):
                break 
        # if self.goal_found:
        #     print(" path successfully found to goal")

        #     # Reconstruct the initial raw Dubins path
        #     self.reconstruct_db_path()

        #     # Then smooth it
        #     smoothed_path = self.smooth_paths()

        #     return smoothed_path, self.get_tree()

        if self.goal_found:
            print(" path successfully found to goal")
            return self.reconstruct_db_path(), self.path    
        return [], self.get_tree()
   