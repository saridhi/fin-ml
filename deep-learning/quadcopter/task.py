import numpy as np
from physics_sim import PhysicsSim

class Task():
    #This Task is defined to land the quadcopter where x=y=z=0 and velocity x=y=z=0
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_v=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 9
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 
        self.target_v = target_v if target_v is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim and velocity to return reward."""
        reward_pos = np.tanh(1.-(0.1*abs(self.sim.pose[:3] - self.target_pos))).sum()
        reward_v = np.tanh(1.-(0.1*abs(self.sim.v[:3] - self.target_v))).sum()
        return (reward_pos + reward_v)/2

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        v_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(np.hstack([self.sim.pose, self.sim.v]))
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #print(self.sim.pose)
        #print(np.hstack([self.sim.pose, self.sim.v]) * self.action_repeat)
        state = np.concatenate([np.hstack([self.sim.pose, self.sim.v])] * self.action_repeat)
        return state