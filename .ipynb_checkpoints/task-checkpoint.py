import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
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

        self.state_size = self.action_repeat * 12
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        
        # a reward of 10 for each successful step
        # discount based of Manhattan Distance between target and current position
        # discount proportionally to sum of Euler angle
        reward = 10 - 0.01*(abs(self.sim.pose[:3] - self.target_pos)).sum() - 0.01*self.sim.pose[3:6].sum()
        
        # different weightage of penalty for gap in x, y, z
        penalty_x = abs(self.sim.pose[0] - self.target_pos[0]) * 0.002
        penalty_y = abs(self.sim.pose[1] - self.target_pos[1]) * 0.002
        penalty_z = abs(self.sim.pose[2] - self.target_pos[2]) * 0.01
        
        reward = reward - penalty_x - penalty_y - penalty_z
        
        # Extra reward if sum of Manhattan Distance is within 30
        if (abs(self.sim.pose[:3] - self.target_pos)).sum() < 30:
             reward += 100

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        all_states = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            all_states.append(np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v]))
        next_state = np.concatenate(all_states)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose, self.sim.v, self.sim.angular_v] * self.action_repeat) 
        return state