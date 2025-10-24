import numpy as np
import random
import math



# samples candidate goals
def sample_candidate_goals(origin, resolution, map_shape, num_samples=200, min_clearance=0.22, bounds=None):
    """Randomly sample points within the known map area and filter close-to-obstacles."""
    goals = []
    if bounds is None:
        min_x = origin[0]
        max_x = origin[0] + map_shape[0] * resolution
        min_y = origin[1]
        max_y = origin[1] + map_shape[1] * resolution
    else:
        min_x, max_x, min_y, max_y = bounds


    for _ in range(num_samples * 5):  # oversamples to improve diversity
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        goals.append([x, y])
        if len(goals) >= num_samples:
            break

    return goals

#estimates how much unknown space is around a sampled goal.
def compute_info_gain(map_data, goal, origin, resolution, radius=2.0):
    """Estimate how much unknown space is visible from the goal pose."""
    info_gain = 0
    radius_cells = int(radius / resolution) # convert radius in meters to map cells
    gx, gy = goal_to_map_coords(goal, origin, resolution)
    #iterates over a square area around the goal
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            mx = gx + dx
            my = gy + dy
            # check if within bounds
            if 0 <= mx < map_data.shape[0] and 0 <= my < map_data.shape[1]:
                #check if cell is unknown, increment info gain
                if map_data[mx, my] == -1:
                    info_gain += 1
    return info_gain * (resolution ** 2)  # in square meters


def goal_to_map_coords(goal, origin, resolution):
    """Convert world coordinates to map grid indices."""
    mx = int((goal[0] - origin[0]) / resolution)
    my = int((goal[1] - origin[1]) / resolution)
    return mx, my

# ranks candidate goals based on their info gain and distance from the robot
def score_goals(goals, robot_pose, info_gains, weight_gain=1.5, weight_dist=0.8):
    """Score goals based on info gain and distance. Filter 0-gain goals before scoring."""
    scored = []

    for i, goal in enumerate(goals):
        if info_gains[i] < 0.01:  # filter zero-gain goals
            continue # skips goals with negligible info gain

        dist = np.linalg.norm(np.array(goal) - np.array(robot_pose[:2]))
        #weighted info gain and distance
        score = weight_gain * info_gains[i] - weight_dist * dist
        scored.append((score, i))
    # if no goals have non zero info gain, return to the first goal
    if not scored:
        return 0  # fallback

    # Return index of goal with best score
    _, best_idx = max(scored, key=lambda x: x[0])
    return best_idx
