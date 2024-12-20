import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob


def process_and_log_multiple_methods(logdirs, output_logdir, tags=["cumulative_reward", "Cumulative Reward"]):

    algorithm_results = {} 

    for logdir in logdirs:
        algorithm_name = logdir.split("/")[-2] 
        if algorithm_name not in algorithm_results:
            algorithm_results[algorithm_name] = []

        event_acc = EventAccumulator(logdir)
        event_acc.Reload()

        available_tags = event_acc.Tags()["scalars"]
        matching_tag = None
        for tag in tags:
            if tag in available_tags:  
                matching_tag = tag
                break

        if not matching_tag:
            print(f"No matching tag found in logdir: {logdir}")
            continue

        scalars = event_acc.Scalars(matching_tag)
        steps = [s.step for s in scalars]
        values = [s.value for s in scalars]
        algorithm_results[algorithm_name].append((steps, values))

    if not algorithm_results:
        print("No valid TensorBoard logs found.")
        return

    line_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 'pink', 'brown']

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (algorithm, data) in enumerate(algorithm_results.items()):
        all_steps = []
        all_values = []

        for steps, values in data:
            all_steps.append(steps)
            all_values.append(values)

        if not all_steps:
            print(f"No valid data for algorithm: {algorithm}")
            continue

        unique_steps = np.unique(np.concatenate(all_steps))
        interpolated_values = [
            np.interp(unique_steps, steps, values) for steps, values in zip(all_steps, all_values)
        ]
        interpolated_values = np.array(interpolated_values)  # shape: (num_episodes, num_steps)

        mean_values = np.mean(interpolated_values, axis=0)
        std_values = np.std(interpolated_values, axis=0)

        color = line_colors[i % len(line_colors)]  
        ax.plot(unique_steps, mean_values, label=f"{algorithm} (Mean)", color=color)
        ax.fill_between(unique_steps, mean_values - std_values, mean_values + std_values,
                        color=color, alpha=0.2)

    ax.set_title("Comparison of Rewards")
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.legend()

    writer = SummaryWriter(output_logdir)
    writer.add_figure("cumulative_reward/comparison_plot", fig)
    writer.close()
    print(f"Comparison graph saved to TensorBoard at {output_logdir}")

logdirs = glob.glob("/home/kyc/2024_RL_Final_Project/MBRL/Dreamerv3/logdir/MFRL_PLOT/SAC_PLOT/**/events.out.tfevents.*", recursive=True)
output_logdir = "/home/kyc/2024_RL_Final_Project/MBRL/Dreamerv3/logdir/tensorboard"
process_and_log_multiple_methods(logdirs, output_logdir, tags=["cumulative_reward", "Cumulative Reward"])
