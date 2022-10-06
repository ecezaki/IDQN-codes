import numpy as np
import math

class LinearController:
    """Linear function of state"""
    def __init__(self, K):
        self.K = K # K has shape action_dim x state_dim

    def get_prob(self, state, action):
        if np.linalg.norm(np.dot(self.K, state)- action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        return np.dot(self.K, state)

class NonLinearController:
    """Non linear controller for swing up"""
    def __init__(self, env):
        self.masscart = env.masscart
        self.masspole = env.masspole
        self.length = env.length
        self.gravity = env.gravity
        self.kv = 40
        self.kx = 1e-2
        

    def get_prob(self, state, action):
        state1 = np.array([state[0], state[2]])
        stateder = np.array([state[1], -state[3]])
        Mq =  np.array([[self.masscart+self.masspole, self.masspole*self.length*math.cos(state[2]) ],[self.masspole*self.length*math.cos(state[2]), (4/3)*self.masspole*(self.length**2)]])
        #E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) - self.masspole*self.gravity*self.length*(math.cos(state[2])-1)
        E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) + self.masspole*self.gravity*self.length*(math.cos(state[2]))
        force = (self.kv*self.masspole*math.sin(state[2])*(self.gravity*math.cos(state[2])-(4/3)*self.length*state[3]**2)   -  ((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(self.kx*state[0]+state[1])  )  /((4/3)*self.kv+((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(E-self.masspole*self.gravity*self.length))
        if np.linalg.norm(force - action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        state1 = np.array([state[0], state[2]])
        stateder = np.array([state[1], -state[3]])
        Mq =  np.array([[self.masscart+self.masspole, self.masspole*self.length*math.cos(state[2]) ],[self.masspole*self.length*math.cos(state[2]), (4/3)*self.masspole*(self.length**2)]])
        #E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) - self.masspole*self.gravity*self.length*(math.cos(state[2])-1)
        E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) + self.masspole*self.gravity*self.length*(math.cos(state[2]))
        force = (self.kv*self.masspole*math.sin(state[2])*(self.gravity*math.cos(state[2])-(4/3)*self.length*state[3]**2)   -  ((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(self.kx*state[0]+state[1])  )  /((4/3)*self.kv+((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(E-self.masspole*self.gravity*self.length))
        return force
    
class NonLinearControllerEnergyShaping:
    """Non linear controller for swing up"""
    def __init__(self, env):
        self.masscart = env.masscart
        self.masspole = env.masspole
        self.length = env.length
        self.gravity = env.gravity
        self.kE = 2.0
        

    def get_prob(self, state, action):
        x = state[0]
        xdot = state[1]
        theta = math.pi-state[2]
        thetadot = -state[3]
        
        Ee = (0.5)*self.masspole*(self.length**2)*thetadot**2 - self.masspole*self.gravity*self.length*(1+math.cos(theta))
        Acc = self.kE*Ee*thetadot*math.cos(theta)
        delta = self.masscart + self.masspole*(math.sin(theta))**2
        
        force = Acc*delta - self.masspole*self.length*(thetadot**2)*math.sin(theta) - self.masspole*self.gravity*math.sin(theta)*math.cos(theta)
        
        # state1 = np.array([state[0], state[2]])
        # stateder = np.array([state[1], state[3]])
        # Mq =  np.array([[self.masscart+self.masspole, self.masspole*self.length*math.cos(state[2]) ],[self.masspole*self.length*math.cos(state[2]), (4/3)*self.masspole*(self.length**2)]])
        # #E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) - self.masspole*self.gravity*self.length*(math.cos(state[2])-1)
        # E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) + self.masspole*self.gravity*self.length*(math.cos(state[2]))
        # force = (self.kv*self.masspole*math.sin(state[2])*(self.gravity*math.cos(state[2])-(4/3)*self.length*state[3]**2)   -  ((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(self.kx*state[0]+state[1])  )  /((4/3)*self.kv+((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(E-self.masspole*self.gravity*self.length))
        if np.linalg.norm(force - action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        x = state[0]
        xdot = state[1]
        theta = math.pi-state[2]
        thetadot = -state[3]
        
        Ee = (0.5)*self.masspole*(self.length**2)*thetadot**2 - self.masspole*self.gravity*self.length*(1+math.cos(theta))
        Acc = self.kE*Ee*thetadot*math.cos(theta)
        delta = self.masscart + self.masspole*(math.sin(theta))**2
        
        force = Acc*delta - self.masspole*self.length*(thetadot**2)*math.sin(theta) - self.masspole*self.gravity*math.sin(theta)*math.cos(theta)
        
        
        # state1 = np.array([state[0], state[2]])
        # stateder = np.array([state[1], state[3]])
        # Mq =  np.array([[self.masscart+self.masspole, self.masspole*self.length*math.cos(state[2]) ],[self.masspole*self.length*math.cos(state[2]), (4/3)*self.masspole*(self.length**2)]])
        # #E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) - self.masspole*self.gravity*self.length*(math.cos(state[2])-1)
        # E = (1/2)*np.dot(np.dot(Mq, stateder.T).T, stateder.T) + self.masspole*self.gravity*self.length*(math.cos(state[2]))
        # force = (self.kv*self.masspole*math.sin(state[2])*(self.gravity*math.cos(state[2])-(4/3)*self.length*state[3]**2)   -  ((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(self.kx*state[0]+state[1])  )  /((4/3)*self.kv+((4/3)*self.masscart+(1/3)*self.masspole+self.masspole*(math.sin(state[2]))**2)*(E-self.masspole*self.gravity*self.length))
        return force


class SimpleNonLinear:
    """Non linear controller for swing up"""
    def __init__(self, env):
        self.masscart = env.masscart
        self.masspole = env.masspole
        self.length = env.length
        self.gravity = env.gravity
        self.C = 1.0
        

    def get_prob(self, state, action):
        x = state[0]
        xdot = state[1]
        theta = math.pi-state[2]
        thetadot = -state[3]
        
        
        force = self.C if thetadot<=0 else -self.C
        
        if np.linalg.norm(force - action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        x = state[0]
        xdot = state[1]
        theta = math.pi-state[2]
        thetadot = -state[3]
        
        
        force = self.C if thetadot<=0 else -self.C
        
        return force


class ApplyBrakesLinearVelocity:
    """Non linear controller for swing up"""
    def __init__(self, env):
        self.masscart = env.masscart
        self.masspole = env.masspole
        self.length = env.length
        self.gravity = env.gravity
        self.C = 5.0
        

    def get_prob(self, state, action):
       
        xdot = state[1]
        
        
        force = self.C if xdot<=0 else -self.C
        
        if np.linalg.norm(force - action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        
        xdot = state[1]
        
        force = self.C if xdot<=0 else -self.C
        
        return force

class ApplyBrakesAngularVelocity:
    """Non linear controller for swing up"""
    def __init__(self, env):
        self.masscart = env.masscart
        self.masspole = env.masspole
        self.length = env.length
        self.gravity = env.gravity
        self.C = 5.0
        

    def get_prob(self, state, action):
       
        thetadot = state[3]
        
        force = -self.C if thetadot<=0 else self.C
        
        if np.linalg.norm(force - action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        
        thetadot = state[3]
        
        force = -self.C if thetadot<=0 else self.C
        
        return force

class FixedActionController:
    """Controller that always plays the same (vector) action irrespective of state"""
    def __init__(self, a):
        self.action = a

    def get_prob(self, state, action):
        if np.linalg.norm(self.action - action) < 1e-8: # arbitrary tolerance
            return 1.0
        else: 
            return 0.0

    def get_action(self, state):
        return self.action
    
