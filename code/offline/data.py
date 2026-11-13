import minari
import numpy as np

from common.replay_buffer import ReplayBuffer


def get_buffer_and_environments_for_task(minari_dataset_name="mujoco/hopper/medium-v0", device="cpu"):
    """
    Loads a Minari dataset, creates an evaluation environment, and populates a
    replay buffer with the dataset's transitions.

    Args:
        minari_dataset_name (str): The name of the Minari dataset to load.
        device (str): The device the buffer returns sampled batches on.

    Returns:
        tuple: (ReplayBuffer populated with the offline data, gym.Env for evaluation)
    """
    dataset = None
    offline_buffer = None
    eval_env = None

    # ------ Problem 2.1: Load the offline dataset into the replay buffer and create the evaluation env ------
    # Reference the Minari documentation: https://minari.farama.org/content/basic_usage/
    #
    # Hints:
    # - minari.load_dataset(..., download=True) gives you episodes that you can iterate over. The replay
    #   buffer stores transitions, so you need to get all the transitions of each episode and add them.
    # - ReplayBuffer(obs_dim, act_dim, max_size, device) has `add_batch(obs, actions, rewards, next_obs, dones)`,
    #   which is much faster than adding transitions one at a time.
    # - Create the evaluation env with render_mode="rgb_array" so that evaluation videos can be recorded.
    # - If you are unsure what fields a Minari object has, put a `breakpoint()` here and inspect it
    #   in pdb with `p dir(object)`.
    ### BEGIN STUDENT SOLUTION - 2.1 ###
    dataset = minari.load_dataset(minari_dataset_name, download=True)
    eval_env = dataset.recover_environment(render_mode="rgb_array")

    obs_dim = int(np.prod(eval_env.observation_space.shape))
    act_dim = int(np.prod(eval_env.action_space.shape))

    # initialize replay buffer
    offline_buffer = ReplayBuffer(obs_dim, act_dim, max_size=dataset.total_steps, device=device)

    for episode in dataset.iterate_episodes():
        # observations has one more entry than actions as it includes the final observation as well
        observation = episode.observations[:-1]
        next_observation = episode.observations[1:]
        # we only use terminations as done
        offline_buffer.add_batch(observation, episode.actions, episode.rewards, next_observation, episode.terminations)
    ### END STUDENT SOLUTION - 2.1 ###

    assert offline_buffer.size == dataset.total_steps, \
        "The buffer size should be equal to the total number of steps in the dataset"
    assert eval_env is not None, "The evaluation environment should not be None"
    return offline_buffer, eval_env
