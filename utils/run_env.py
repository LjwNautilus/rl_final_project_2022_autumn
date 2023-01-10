import torch

from vmas.simulator.utils import save_video

def run_env(
    env,
    policy,
    n_steps=1000,
    render=True,
    save_render=False,
    save_name='video',
    centralised=False,
):
    assert not (save_render and not render), \
        "To save the video you have to render it"

    frame_list = []  # For creating a gif
    step = 0
    obs = env.reset()
    total_reward = 0
    for _ in range(n_steps):
        step += 1
        if centralised:
            if isinstance(obs, list):
                obs = torch.concat(obs, dim=1)
            actions = \
                policy.compute_action(obs, u_range=env.agents[0].u_range)
            action_list = actions.split(actions.shape[1] // env.n_agents, dim=1)
            obs, rews, dones, _ = env.step(action_list)
        else:
            actions = [None] * len(obs)
            for i in range(len(obs)):
                actions[i] = \
                    policy.compute_action(obs[i], u_range=env.agents[i].u_range)
            obs, rews, dones, _ = env.step(actions)
        rewards = torch.stack(rews, dim=1)
        global_reward = rewards.mean(dim=1)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward
        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )
        if dones.item():
            break
    if render and save_render:
        save_video(save_name, frame_list, 1 / env.scenario.world.dt)
    return total_reward.item()