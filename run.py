# Import necessary modules
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Import classes from TensorForce library
from tensorforce import TensorForceError
from tensorforce.agents import Agent
from tensorforce.execution import Runner

# Import custom environment from src folder
from src.environment import Optimization_of_Join_Order_Sequence_Pattern

# Import other required libraries
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import sys
import os
import json


# Function to create command-line arguments
def make_args_parser():
    parser = argparse.ArgumentParser()

    # Add arguments for agent configuration file, network specification file, number of episodes, groups, target group,
    # incremental mode, maximum number of timesteps per episode, specific query to run, save agent to directory,
    # restore agent from directory, output directory, testing mode, order queries by relations_num, save agent every x episodes, and select phase
    parser.add_argument(
        "-a",
        "--agent-config",
        default="config/ppo.json",
        help="Agent configuration file",
    )
    parser.add_argument(
        "-n",
        "--network-spec",
        default="config/complex-network.json",
        help="Network specification file",
    )
    parser.add_argument(
        "-e", "--episodes", type=int, default=800, help="Number of episodes"
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=int,
        default=1,
        help="Total groups of different number of relations",
    )
    parser.add_argument(
        "-tg", "--target_group", type=int, default=4, help="A specific group"
    )
    parser.add_argument(
        "-m", "--mode", type=str, default="round", help="Incremental Mode"
    )
    parser.add_argument(
        "-ti",
        "--max-timesteps",
        type=int,
        default=20,
        help="Maximum number of timesteps per episode",
    )
    parser.add_argument("-q", "--query", default="", help="Run specific query")
    parser.add_argument("-s", "--save_agent", default="", help="Save agent to this dir")
    parser.add_argument("-r", "--restore_agent", default="", help="Restore Agent from this dir")
    parser.add_argument("-o", "--outputs", default="./outputs/", help="Restore Agent from this dir")
    parser.add_argument(
        "-t",
        "--testing",
        action="store_true",
        default=False,
        help="Test agent without learning.",
    )
    parser.add_argument('-all', '--run_all', action='store_true', default=False, help="Order queries by relations_num")
    parser.add_argument(
        "-se",
        "--save_episodes",
        type=int,
        default=100,
        help="Save agent every x episodes",
    )
    parser.add_argument("-p", "--phase", help="Select phase (1 or 2)", default=1)

    return parser.parse_args()


# Function to print out the configuration settings
def print_config(args):
    print("Running with the following configuration")
    arg_map = vars(args)
    for key in arg_map:
        print("\t", key, "->", arg_map[key])

# Main function
def main():

    # Parse command-line arguments
    args = make_args_parser()
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ### ************************************************************************************************************************************** ###

    ##########################################################################
    ########################## Setting up the Model ##########################
    ##########################################################################

    # Set up environment
    memory = {}
    environment = Optimization_of_Join_Order_Sequence_Pattern(
        args.phase,
        args.query,
        args.episodes,
        args.groups,
        memory,
        args.mode,
        args.target_group,
        args.run_all
    )

    # Load agent configuration from file
    if args.agent_config is not None:
        with open(args.agent_config, "r") as fp:
            agent_config = json.load(fp=fp)
    else:
        print('Error')
        raise TensorForceError("No agent configuration provided.")

    # Load network specification from file
    if args.network_spec is not None:
        with open(args.network_spec, "r") as fp:
            network_spec = json.load(fp=fp)
    else:
        print('Error')
        raise TensorForceError("No network configuration provided.")

    # Set up the PPO Agent
    agent = Agent.from_spec(
        spec=agent_config,
        kwargs=dict(
            states=environment.states, actions=environment.actions, network=network_spec,
            variable_noise=0.5
        ),
    )

    # Restore agent from directory if specified
    if args.restore_agent != "":
        agent.restore_model(directory=args.restore_agent)

    # Set up runner
    runner = Runner(agent=agent, environment=environment)
    ############################## Finished ##################################

    ### ************************************************************************************************************************************** ###

    ##########################################################################
    ############################# Save the Model #############################
    ##########################################################################
    report_episodes = 1 # Number of reports to generate during training/testing

    
    def episode_finished(r):
        # Check if it is time to generate a report
        if r.episode % report_episodes == 0:
            if args.save_agent != "" and args.testing is False and r.episode == (args.episodes + args.save_episodes):
                save_dir = os.path.dirname(args.save_agent)
                if os.path.isdir(save_dir):
                    try:
                        os.mkdir(save_dir, 0o755)
                    except OSError:
                        raise OSError("Cannot save agent to dir {} ()".format(save_dir))
                    
                # Save the trained model
                r.agent.save_model(
                    directory=args.save_agent, append_timestep=True
                )

            # Print out the episode reward and the average reward over the last 100 episodes
            logger.info(
                "Episode {ep} reward: {r}".format(ep=r.episode, r=r.episode_rewards[-1])
            )
            logger.info(
                "Average of last 100 rewards: {}\n".format(
                    sum(r.episode_rewards[-100:]) / 100
                )
            )
        return True
    ############################## Finished ##################################

    ### ************************************************************************************************************************************** ###

    ##########################################################################
    ######################## Start training or testing #######################
    ##########################################################################
    logger.info(
        "Starting {agent} for Environment '{env}'".format(agent=agent, env=environment)
    )

    # Run the training/testing loop 
    runner.run(
        episodes=args.episodes,
        max_episode_timesteps=args.max_timesteps,
        episode_finished=episode_finished,
        deterministic=args.testing,
    )

    runner.close()

    # Print out the total number of episodes
    logger.info("Learning finished. Total episodes: {ep}".format(ep=runner.episode))
    ############################## Finished ##################################

    ### ************************************************************************************************************************************** ###

    ##########################################################################
    ############################ Find convergence ############################
    ##########################################################################
    def find_convergence(eps):
        last = eps[-1]
        for i in range(1, len(eps)):
            # Check if the episode rewards have converged
            if eps[i * -1] != last:
                print("Converged at episode:", len(eps) - i + 2)
                return True
    
    # Find the convergence point of the episode rewards
    find_convergence(runner.episode_rewards)
    ############################## Finished ##################################

    ### ************************************************************************************************************************************** ###

    ##########################################################################
    ################# Plot recorded costs over all episodes ##################
    ##########################################################################

    # Plot histogram of episode rewards
    plt.figure(figsize=(8, 6))
    plt.hist(runner.episode_rewards, color="mediumseagreen", alpha=0.7)
    plt.title("Histogram of Episode Rewards")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(args.outputs + "episode_rewards_histogram.png")

    # Plot episode rewards over time
    plt.figure(figsize=(8, 6))
    plt.plot(runner.episode_rewards, "b.")
    plt.title("Episode Rewards over Time")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(args.outputs + "episode_rewards_over_time.png")

    # Plot costs for each query
    if not os.path.exists(args.outputs):
        os.makedirs(args.outputs)

    # Plot for each query
    for i, (file, val) in enumerate(memory.items()):
        costs = np.array(val["costs"])
        postgres_estimate = val["postgres_cost"]
        min_val = min(costs)
        max_val = max(costs)

        plt.figure(figsize=(8, 6))
        plt.title(file)
        plt.xlabel("Episode")
        plt.ylabel("Cost")

        plt.scatter(
            np.arange(len(costs)),
            costs,
            c="mediumseagreen",
            alpha=0.5,
            marker=r"$\ast$",
            label="Cost",
        )

        plt.scatter(
            0,
            [min_val],
            c="tomato",
            alpha=1,
            marker=r"$\heartsuit$",
            s=200,
            label="Min cost observed=" + str(min_val),
        )

        plt.scatter(
            0,
            [max_val],
            c="dodgerblue",
            alpha=1,
            marker=r"$\times$",
            s=200,
            label="Max cost observed=" + str(max_val),
        )

        plt.scatter(
            0,
            [postgres_estimate],
            c="purple",
            alpha=1,
            marker=r"$\star$",
            s=200,
            label="PostgreSQL estimate=" + str(postgres_estimate),
        )

        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(args.outputs + file + ".png")

    plt.show(block=True)

    ############################## Finished ##################################

if __name__ == "__main__":
    main()
