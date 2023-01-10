#  Copyright (c) 2022.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color

class Scenario(BaseScenario):
    '''Comments by 20337087 Jingwu Luo

    In the transport senario, `n_agents` agent(s) is going to push
    `n_packages` (default 1) package(s) to a goal.
    Agents, packages and the goal are spawned
    at random positions between -1 and 1.
    When all packages overlap with the goal, the senario ends.

    Packages are boxes with `package_mass` mass (default 50 times agent mass)
    and `package_width` and `package_length` as sizes.

    Each agent observes its position, velocity, relative position to packages,
    package velocities, relative positions between packages
    and the goal and a flag for each package indicating if it is on the goal.

    pushing a package towards the goal will give a positive reward,
    while pushing it away, a negative one.
    Agents need to collaborate and push packages together
    to be able to move them faster.
    '''

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        '''Sets basic elements of the interactive environment.'''

        # Reads environment-based arguments from kwargs.
        n_agents = kwargs.get("n_agents", 4)  # Number of agents
        # Number of packages to be pushed by agents.
        self.n_packages = kwargs.get("n_packages", 1)
        # Width and length of the package relative to size of the environment.
        # The possible positions of the package are between -1 and 1
        # so the environment is of size [2, 2] by default.
        self.package_width = kwargs.get("package_width", 0.15)
        self.package_length = kwargs.get("package_length", 0.15)
        # Mass of the package, 50 times of agents' mass by default.
        self.package_mass = kwargs.get("package_mass", 50)

        self.shaping_factor = 100

        # Make world
        world = World(batch_dim, device)
        # Add agent(s) to the environment.
        for i in range(n_agents):
            agent = Agent(name=f"agent {i}", shape=Sphere(0.03), u_multiplier=0.6)
            world.add_agent(agent)
        # Add the goal of package(s).
        goal = Landmark(
            name="goal",
            # The goal is not able to collide with agents and packages.
            collide=False,  
            # Shape of landmarks is a sphere/circle by default.
            shape=Sphere(radius=0.15),  
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        # Adds package(s) to the environment.
        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package {i}",
                collide=True,
                movable=True,
                mass=50,
                # Shape of landmarks is a box by default.
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)

        return world

    def reset_world_at(self, env_index: int = None):
        '''Resets the environment whose index is `env_index`.'''

        # The goal is the first landmark in the environment.
        goal = self.world.landmarks[0]
        
        goal.set_pos(
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
        # Randomly sets position of package(s) between -1 and 1.
        for i, package in enumerate(self.packages):
            package.set_pos(
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
            package.on_goal = self.world.is_overlapping(package, package.goal)
            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )
        # Randomly sets position of agent(s) between -1 and 1.
        for i, agent in enumerate(self.world.agents):
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
            # If one of the agents overlaps with one of the packages,
            # resets position of the overlapped agent(s).
            for package in self.packages:
                while self.world.is_overlapping(
                    agent, package, env_index=env_index
                ).any():
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

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]
        # Only recalculates rewards for the first agent.
        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            for i, package in enumerate(self.packages):
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                # Packages which do not overlap with the goal are drawn red.
                package.color = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                ).repeat(self.world.batch_dim, 1)
                # When a package overlaps with the goal,
                # its color changes green.
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )
                # When a package overlaps with the goal,
                # its contribution to reward becomes 0.
                # If a package comes closer to the goal,
                # the reward would be positive otherwise would be negative.
                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        package_obs = []
        for package in self.packages:
            # Position of packages relative to the goal
            package_obs.append(package.state.pos - package.goal.state.pos)
            # Relative position of agents to packages
            package_obs.append(package.state.pos - agent.state.pos)
            # Velocities of packages
            package_obs.append(package.state.vel)
            # Which package is on the goal.
            package_obs.append(package.on_goal.unsqueeze(-1))

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *package_obs,
            ],
            dim=-1,
        )

    def done(self):
        # When all packages overlap with the goal, the scenario is done.
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )