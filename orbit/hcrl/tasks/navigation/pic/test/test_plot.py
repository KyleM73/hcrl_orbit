import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append("..")
from pic import PathIntegralController

def fit_to_grid(x, border_radius=3, s=10):
    return int(x * s) + int(border_radius * s)

def fit_to_grid_vec(x, border_radius=3, s=10):
    return (s * x).to(int) + int(border_radius * s)

def test_single_plot(file_path: str = "test_single.mp4", grid_scale: int = 20):
     # parameters for all tests
    dt, T = 0.1, 10
    num_samples = 10000
    border_radius = 4
    box_radius = 0.5
    device = "cpu"
    obs = torch.tensor([
        3, 3, 0,   # pose
        0,         # heading
        0, 0, 0,   # lin vel
        0, 0, 0,   # ang vel
        1.5, 1.5, 0,   # box pose
    ]).view(1, -1).to(dtype=torch.float32, device=device)
    policy = PathIntegralController(obs, dt, T, num_samples, border_radius, box_radius, device=device)

    fig, ax = plt.subplots()
    ax.axis("off")
    image_list = []
    traj_hist = []
    collision_flag = 1.0
    for i in range(int(T/dt)-1):
        image = torch.zeros(2*border_radius*grid_scale+1, 2*border_radius*grid_scale+1, 3)
        x, v_cmd, samples = policy(obs)
        collision_flag = policy.check_collision(obs, x[:, 0, :], x[:, 1, :], collision_flag)
        sample_grid = fit_to_grid_vec(samples, border_radius, grid_scale)
        image[sample_grid[:, :, 0], sample_grid[:, :, 1], 0] = 1
        x_grid = fit_to_grid_vec(x, border_radius, grid_scale)
        x_grid_idx = x_grid[:, 0], x_grid[:, 1], 2
        traj_hist.append(x_grid_idx)
        for x_grid in traj_hist:
            image[*x_grid] = 1
        box_coords = policy.get_box_points(obs)
        for box in box_coords:
            lower_l, upper_l, lower_r, upper_r = box
            x_min, x_max = fit_to_grid(lower_l[0, 0], border_radius, grid_scale), fit_to_grid(lower_r[0, 0], border_radius, grid_scale)
            x_range = torch.arange(x_min, x_max+1)
            y_min, y_max = fit_to_grid(lower_l[0, 1], border_radius, grid_scale), fit_to_grid(upper_l[0, 1], border_radius, grid_scale)
            y_range = torch.arange(y_min, y_max+1)
            image[x_range, y_min, 1] = 1
            image[x_range, y_max, 1] = 1
            image[x_min, y_range, 1] = 1
            image[x_max, y_range, 1] = 1
        image_list.append([ax.imshow(np.flip(image.numpy(), axis=0), animated=True)])
        if collision_flag < 0.9:
            print("Collision!")
            break
        print(i,"/",int(T/dt)-1)
    print("Final State: ",x[0, ...])
    print("Error: ",torch.norm(x[:, :2]))

    ani = animation.ArtistAnimation(fig, image_list, interval=100, blit=True)
    ani.save(file_path)
    print("Movie saved to ",file_path,"\n")

def test_multi_plot(num_trials: int, file_path: str = "test_multi.mp4", grid_scale: int = 20):
    # parameters for all tests
    dt, T = 0.1, 10
    num_samples = 10000
    border_radius = 4
    box_radius = 0.5
    device = "cpu"
    obs = torch.tensor([
        3, 3, 0,   # pose
        0,         # heading
        0, 0, 0,   # lin vel
        0, 0, 0,   # ang vel
        1.5, 1.5, 0,   # box pose
    ]).view(1, -1).to(dtype=torch.float32, device=device)

    fig, ax = plt.subplots()
    ax.axis("off")
    image_list = []
    image_init = torch.zeros(2*border_radius*grid_scale+1, 2*border_radius*grid_scale+1, 3)
    failures = 0
    for k in range(num_trials):
        policy = PathIntegralController(obs, dt, T, num_samples, border_radius, box_radius, device=device)
        traj_hist = []
        collision_flag = 1.0
        image = image_init
        for i in range(int(T/dt)-1):
            x, v_cmd, samples = policy(obs)
            collision_flag = policy.check_collision(obs, x[:, 0, :], x[:, 1, :], collision_flag)
            x_grid = fit_to_grid_vec(x, border_radius, grid_scale)
            x_grid_idx = x_grid[:, 0], x_grid[:, 1]
            traj_hist.append(x_grid_idx)
            if collision_flag < 0.9:
                failures += 1
                break
        color = 2 if collision_flag > 0.9 else 0
        for x_grid in traj_hist:
            image[*x_grid, color] = 1
        if k == 0:
            box_coords = policy.get_box_points(obs)
            for box in box_coords:
                lower_l, upper_l, lower_r, upper_r = box
                x_min, x_max = fit_to_grid(lower_l[0, 0], border_radius, grid_scale), fit_to_grid(lower_r[0, 0], border_radius, grid_scale)
                x_range = torch.arange(x_min, x_max+1)
                y_min, y_max = fit_to_grid(lower_l[0, 1], border_radius, grid_scale), fit_to_grid(upper_l[0, 1], border_radius, grid_scale)
                y_range = torch.arange(y_min, y_max+1)
                image[x_range, y_min, 1] = 1
                image[x_range, y_max, 1] = 1
                image[x_min, y_range, 1] = 1
                image[x_max, y_range, 1] = 1
        image_list.append([ax.imshow(np.flip(image.numpy(), axis=0), animated=True)])
        image_init = image
        print(k,"/",num_trials-1)
    print("Success rate: ",1 - failures/num_trials)
    ani = animation.ArtistAnimation(fig, image_list, interval=100, blit=True)
    ani.save(file_path)
    print("Movie saved to ",file_path,"\n")

if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    test_single_plot(dir+"/test_single.mp4", 20)
    test_multi_plot(100, dir+"/test_multi.mp4", 20)

