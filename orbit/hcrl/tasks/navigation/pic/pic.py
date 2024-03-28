from typing import Tuple, List

import torch
torch.manual_seed(0)

class PathIntegralController:
    def __init__(self,
                 obs: torch.Tensor,
                 dt: float,
                 T: float,
                 num_samples: int, 
                 border_radius: float = 5.0,
                 box_radius: float = 0.5,
                 device: str = "cpu",
                 ) -> None:
        # simulation parameters
        self.dt = dt
        self.T = T
        self.num_steps = int(T/dt)
        self.num_samples = num_samples
        self.border_radius = border_radius
        self.box_radius = box_radius
        self.device = device

        # tuning parameters
        self.s2 = 0.01
        self.a, self.b, self.d, self.e, self.eta = 1.0, 1.0, 1.0, 2.0, 1.0
        self.lambd = self.a * self.s2
        self.k1 = -self.e / self.T
        self.k2, self.k3 = self.k1, self.k1
        self.s = self.s2**0.5
        self.G_u = torch.tensor([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=torch.float32, device=self.device)

        # tracking objects
        self.reset(obs)

    def __call__(self, obs: torch.Tensor) -> Tuple[torch.Tensor]:
        assert self.step != self.num_steps+1

        self.x_sampled[self.step] = self.x
        self.S_tau[:] = 0
        self.collision_flag[:] = 1.0
        
        for i in range(self.step+1, self.num_steps):
            self.S_tau += self.collision_flag * self.dt * self.b * torch.square(torch.norm(self.x_sampled[i, :, :2], dim=1))
            noisy_control = torch.einsum("ij,ajk->aik", self.G_u, self.s * self.eps_sampled[i] * self.dt**0.5)
            self.x_sampled[i+1] = self.x_sampled[i] + (self.f(self.x_sampled[i]) * self.dt + noisy_control) * self.collision_flag.repeat(1, 4).unsqueeze(-1)
            self.collision_flag, self.S_tau = self.check_collision(
                obs, self.x_sampled[i+1, :, 0], self.x_sampled[i+1, :, 1], self.collision_flag, self.S_tau)
        self.S_tau += torch.where(self.collision_flag > 0.9, self.d * torch.square(torch.norm(self.x_sampled[self.num_steps, :, :2], dim=1)), 0.0)

        denom_i = torch.exp(-self.S_tau / self.lambd)
        numer = torch.einsum("ijk,ik->jk",self.eps_sampled[self.step], denom_i)
        denom = torch.sum(denom_i)

        u = self.s / self.dt**0.5 * numer / denom # v, theta
        noisy_control = torch.einsum("ij,jk->ik", self.G_u, u * self.dt + self.s * self.eps[self.step] * self.dt**0.5)
        self.x = self.x + self.f(self.x) * self.dt + noisy_control

        self.step += 1

        return self.x, u, self.x_sampled[self.step-1:]

    def reset(self, obs: torch.Tensor) -> None:
        self.x = self.get_state(obs) # x1, x2, v, theta
        self.eps = torch.randn(self.num_steps+1, 2, 1, dtype=torch.float32, device=self.device)
        self.x_sampled = torch.zeros(self.num_steps+1, self.num_samples, 4, 1, dtype=torch.float32, device=self.device)
        self.f_const = torch.diag(torch.tensor([self.k1, self.k1, self.k2, self.k3], dtype=torch.float32, device=self.device))
        self.eps_sampled = torch.randn(self.num_steps+1, self.num_samples, 2, 1, dtype=torch.float32, device=self.device)
        self.S_tau = torch.zeros(self.num_samples, 1, dtype=torch.float32, device=self.device)
        self.collision_flag = torch.ones(self.num_samples, 1, dtype=torch.float32, device=self.device)
        self.step = 0

    def f(self, x: torch.Tensor) -> torch.Tensor:
        f_non_lin = torch.zeros_like(self.f_const.expand(x.size(0), 4, 4))
        f_non_lin[:, 0, 2], f_non_lin[:, 1, 2] = torch.cos(x[..., 3, :]).squeeze(), torch.sin(x[..., 3, :]).squeeze()
        f_full = self.f_const + f_non_lin
        return torch.einsum("aij,ajk->aik", f_full, x)

    def get_state(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[obs[:, 0]], [obs[:, 1]], [obs[:, 4:6].norm(dim=1)], [obs[:, 3]]], dtype=torch.float32, device=self.device).view(-1, 4, 1)
    
    def get_box_points(self, obs: torch.Tensor) -> List[torch.Tensor]:
        assert obs.size(1) > 10

        box_poses = obs[:, 10:]
        box_coords = [[[box_poses[:, 3*i:3*i+2] + torch.tensor([[j, k]], device=self.device) 
                        for k in [-self.box_radius, self.box_radius]]
                    for j in [-self.box_radius, self.box_radius]] 
                for i in range(box_poses.size(1) // 3)]
        return [[*box_coords[i][0], *box_coords[i][1]] for i in range(len(box_coords))]
    
    def check_collision(self,
                        obs: torch.Tensor,
                        x: torch.Tensor,
                        y: torch.Tensor,
                        collision_flag: torch.Tensor,
                        S_tau: torch.Tensor,
                        ) -> Tuple[torch.Tensor]:
        # no penalty if already collided on previous iteration
        penalty = self.eta * (collision_flag < 0.9)
        # check in bounds
        inside_border = (
            torch.where((-self.border_radius < x) & ( x < self.border_radius), True, False)) & \
            torch.where((-self.border_radius < y) & (y < self.border_radius), True, False)
        box_coords = self.get_box_points(obs)
        outside_boxes_list = [
            ((x < coords[0][0, 0]) & (x < coords[1][0, 0]) | (x > coords[2][0, 0]) & (x > coords[3][0, 0])) &
            ((y < coords[0][0, 1]) | (y > coords[1][0, 1]) & (y > coords[2][0, 1]) | (y < coords[3][0, 1]))
                for coords in box_coords]
        outside_boxes = torch.ones_like(inside_border)
        for cond in outside_boxes_list:
            outside_boxes *= cond
        # set collision flag
        collision_flag = torch.where((collision_flag < 0.9) | ~inside_border | ~outside_boxes, 0.0, collision_flag)
        # collision penalty
        S_tau += torch.where(collision_flag < 0.9, penalty, 0.0)
        return collision_flag, S_tau

if __name__ == "__main__":
    device = "cpu" # ["cpu", "cuda", "mps"]
    obs = torch.tensor([
        4, 4, 0,   # pose
        0,         # heading
        -1, -1, 0, # lin vel
        0, 0, 0,   # ang vel
        -2, -2, 0,   # box pose
        -2, -2, 0, # box pose
    ]).view(1, -1).to(dtype=torch.float32, device=device)
    dt, T = 0.01, 10
    num_samples = 1000

    policy = PathIntegralController(obs, dt, T, num_samples, device=device)
    for i in range(int(T/dt)-1):
        x, v_cmd, samples = policy(obs)
        print(x)
    print("Error: ",torch.norm(x[:, :2]))