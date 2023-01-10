#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from vmas.simulator.core import Agent, Landmark, Sphere, World, Line, Box
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y

class Scenario(BaseScenario):
    '''Comments by 20337087 Jingwu Luo
    
    In the balance senario, particular amount of agents 
    have to carry the package to the goal 
    and try to avoid falling of the package and the line. 
    
    Each agent receives the same reward.
    Getting the package closer to the goal will give a positive reward 
    while moving it away will give a negative one. 
    
    When the package or the line fall, the environment is done 
    and agents get a huge negative reward.
    '''

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Reads environment-based arguments from kwargs.
        self.n_agents = kwargs.get("n_agents", 3)  # number of agents
        # Mass of the package to be pushed,
        # it is 5 times of the mass of an agent by default.
        self.package_mass = kwargs.get("package_mass", 5)
        # Sets whether the package is on a random position of the line.
        self.random_package_pos_on_line = kwargs.get("random_package_pos_on_line", True)
        # This scenario cannot be done by only 1 agent.
        # More than 2 agents are required.
        assert self.n_agents > 1

        self.line_length = 0.8  # Length of the line
        self.agent_radius = 0.03  # Radius of the agent in sphere shape

        self.shaping_factor = 100
        # The penalty of making the package or the line fall to the floor
        self.fall_reward = -10

        # Make world
        # Vertical gravity is set.
        world = World(batch_dim, device, gravity=(0.0, -0.05), y_semidim=1)
        # Add agents
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent {i}", shape=Sphere(self.agent_radius), u_multiplier=0.7
            )
            world.add_agent(agent)
        # Adds the goal to the environment.
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        # Adds the package to the environment.
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # Add landmarks
        # Adds the line to the environment.
        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)
        # Adds the floor to the environment.
        floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(floor)
        # Initializes rewards tensors.
        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        # Resets position of the goal.
        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        # Resets position of the line.
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.agent_radius * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        # Resets position of the package relative to the line.
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.line_length / 2 + self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                    self.line_length / 2 - self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        # Resets positions of agents.
        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -self.agent_radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )
        floor = self.world.landmarks[3]
        floor.set_pos(
            torch.tensor(
                [0, -self.world.y_semidim - floor.shape.width / 2 - self.agent_radius],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0
            # Calculates whether the package or the line is on the ground.
            self.on_the_ground = (
                self.package.state.pos[:, Y] <= -self.world.y_semidim
            ) + (self.line.state.pos[:, Y] <= -self.world.y_semidim)
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )
            # If the package or the line fall into the ground,
            # agents get a fall penalty.
            self.ground_rew[self.on_the_ground] = self.fall_reward

            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

        return self.ground_rew + self.pos_rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        return torch.cat(
            [
                agent.state.pos,  # positions of agents
                agent.state.vel,  # velocities of agents
                # relative positions of agents to the package
                agent.state.pos - self.package.state.pos,
                # relative positions of agents to the line
                agent.state.pos - self.line.state.pos,
                # relative positions of the package to the goal
                self.package.state.pos - self.package.goal.state.pos,
                # velocity of the package
                self.package.state.vel,
                # velocity of the line
                self.line.state.vel,
                # angle velocity of the line
                self.line.state.ang_vel,
                # rotation angle of the line module pi
                self.line.state.rot % torch.pi,
            ],
            dim=-1,
        )

    def done(self):
        # If the package or the line falls into the ground
        # or the package overlaps with the goal,
        # the scenario is done.
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        info = {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew}
        # When reset is called before reward()
        return info