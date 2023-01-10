#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import Agent, Landmark, World, Line, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

class Scenario(BaseScenario):
    '''Comments by 20337087 Jingwu Luo
    
    In the wheel senario, particular amount of agents have to push a line
    to rotate and make the angle velocity of line match a desired value. 
    The current angle velocity gets closer to the desired value, 
    agents get greater rewards.
    '''
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Reads environment-based arguments from kwargs.
        n_agents = kwargs.get("n_agents", 4)  # Number of agents
        # Length of line to be pushed by agents
        self.line_length = kwargs.get("line_length", 2)
        # Mass of the line, it is 30 times of mass of an agent by default.
        line_mass = kwargs.get("line_mass", 30)
        # The desired velocity to be matched.
        self.desired_velocity = kwargs.get("desired_velocity", 0.05)

        # Make world
        world = World(batch_dim, device)
        # Add agents
        for i in range(n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(name=f"agent {i}", u_multiplier=0.6, shape=Sphere(0.03))
            world.add_agent(agent)
        # Add landmarks
        # The line to be pushed.
        self.line = Landmark(
            name="line",
            collide=True,
            rotatable=True,
            shape=Line(length=self.line_length),
            mass=line_mass,
            color=Color.BLACK,
        )
        # The constrained original point.
        world.add_landmark(self.line)
        center = Landmark(
            name="center",
            shape=Sphere(radius=0.02),
            collide=False,
            color=Color.BLACK,
        )
        world.add_landmark(center)

        return world

    def reset_world_at(self, env_index: int = None):
        # Resets positions of agents.
        for agent in self.world.agents:
            # Random pos between -1 and 1
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )
        # Resets states of the line.
        self.line.set_rot(
            torch.zeros(
                (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -torch.pi / 2,
                torch.pi / 2,
            ),
            batch_index=env_index,
        )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        # The closer the current angle velocity to the desired velocity,
        # the greater the rewards are.
        if is_first:
            self.rew = (self.line.state.ang_vel.abs() - self.desired_velocity).abs()

        return -self.rew

    def observation(self, agent: Agent):
        # Positions of two ends of the line.
        line_end_1 = torch.cat(
            [
                (self.line_length / 2) * torch.cos(self.line.state.rot),
                (self.line_length / 2) * torch.sin(self.line.state.rot),
            ],
            dim=1,
        )
        line_end_2 = -line_end_1

        return torch.cat(
            [
                agent.state.pos,  # position of agents
                agent.state.vel,  # current velocity of agents
                # The relative position of agents to the line.
                self.line.state.pos - agent.state.pos,
                # The relative position of agents to two ends of the line.
                line_end_1 - agent.state.pos,
                line_end_2 - agent.state.pos,
                # Current angle of the line module pi
                self.line.state.rot % torch.pi,
                # Current angle velocity of the line
                self.line.state.ang_vel.abs(),
                # The absolute difference between 
                # the current angular velocity of the line and the desired one.
                (self.line.state.ang_vel.abs() - self.desired_velocity).abs(),
            ],
            dim=-1,
        )