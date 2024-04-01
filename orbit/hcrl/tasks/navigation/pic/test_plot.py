import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pic2 import PathIntegralController

def fit_to_grid(x, border_radius=3):
    return int(x * 10) + int(border_radius * 10)

def test_plot():
     # parameters for all tests
    dt, T = 0.1, 10
    num_samples = 10000
    border_radius = 4
    box_radius = 0.5
    device = "cpu"
    num_trials = 1
    obs = torch.tensor([
        3, 3, 0,   # pose
        0,         # heading
        0, 0, 0,   # lin vel
        0, 0, 0,   # ang vel
        1.5, 1.5, 0,   # box pose
    ]).view(1, -1).to(dtype=torch.float64, device=device)
    policy = PathIntegralController(obs, dt, T, num_samples, border_radius, box_radius, device=device)

    fig, ax = plt.subplots()
    ax.axis("off")
    image_list = []
    traj_hist = []
    collision_flag = 1.0
    for i in range(int(T/dt)-1):
        image = torch.zeros(2*border_radius*10+1, 2*border_radius*10+1, 3)
        x, v_cmd, samples = policy(obs)
        collision_flag = policy.check_collision(obs, x[:, 0, :], x[:, 1, :], collision_flag)
        for k in range(samples.size(0)):
            for j in range(samples.size(1)):
                sample_grid = fit_to_grid(samples[k, j, 0], border_radius), fit_to_grid(samples[k, j, 1], border_radius), 0
                image[*sample_grid] = 1
        x_grid = fit_to_grid(x[0, 0], border_radius), fit_to_grid(x[0, 1], border_radius), 2
        traj_hist.append(x_grid)
        for x_grid in traj_hist:
            image[*x_grid] = 1
        c1, c2 = fit_to_grid(1, border_radius), fit_to_grid(2, border_radius)
        for t in range(c1, c2):
            image[t, c2-1, 1] = 1
            image[t, c1-1, 1] = 1
            image[c1-1, t, 1] = 1
            image[c2-1, t, 1] = 1 
        image_list.append([ax.imshow(np.flip(image.numpy(), axis=0), animated=True)])
        if collision_flag < 0.9:
            print("collision")
            break
        print(i,"/",int(T/dt)-1)
    print("Final State: ",x[0, ...])
    print("Error: ",torch.norm(x[:, :2]))

    ani = animation.ArtistAnimation(fig, image_list, interval=100, blit=True)
    ani.save("movie_eta1.mp4")

if __name__ == "__main__":
    test_plot()

