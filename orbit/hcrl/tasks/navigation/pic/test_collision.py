import torch

from pic import PathIntegralController

def test_collision_check():
    # parameters for all tests
    dt, T = 0.01, 1
    num_samples = 1
    border_radius = 5
    box_radius = 0.5
    device = "cpu"
    obs = torch.tensor([
        4, 4, 0,   # pose
        0,         # heading
        0, 0, 0,   # lin vel
        0, 0, 0,   # ang vel
        2, 2, 0,   # box pose
    ]).view(1, -1).to(dtype=torch.float64, device=device)
    policy = PathIntegralController(obs, dt, T, num_samples, border_radius, box_radius, device=device)

    # test no collision
    collision_flag = 1.0
    x = torch.tensor([0])
    y = torch.tensor([0])
    assert policy.check_collision(obs, x, y, collision_flag) == 1

    collision_flag = 1.0
    x = torch.tensor([2.81])
    y = torch.tensor([2.49])
    assert policy.check_collision(obs, x, y, collision_flag) == 1

    collision_flag = 1.0
    x = torch.tensor([2.49])
    y = torch.tensor([2.81])
    assert policy.check_collision(obs, x, y, collision_flag) == 1

    # test collision: border: x
    collision_flag = 1.0
    x = torch.tensor([5])
    y = torch.tensor([0])
    assert policy.check_collision(obs, x, y, collision_flag) == 0

    # test collision: border: y
    collision_flag = 1.0
    x = torch.tensor([0])
    y = torch.tensor([5])
    assert policy.check_collision(obs, x, y, collision_flag) == 0

    # test collision: box: x
    collision_flag = 1.0
    x = torch.tensor([2.5])
    y = torch.tensor([2])
    assert policy.check_collision(obs, x, y, collision_flag) == 0

    # test collision: box: y
    collision_flag = 1.0
    x = torch.tensor([2])
    y = torch.tensor([2.5])
    assert policy.check_collision(obs, x, y, collision_flag) == 0

if __name__ == "__main__":
    test_collision_check()