from utils import get_dataset

u_ref, v_ref, w_ref, p_ref, t_star, x_star, y_star, nu = get_dataset()

print(u_ref.shape)

# # Remove the last time step
# u_ref = u_ref[:-1, :]
# v_ref = v_ref[:-1, :]
# w_ref = w_ref[:-1, :]
# p_ref = p_ref[:-1, :]
u0 = u_ref[0, :]
v0 = v_ref[0, :]
w0 = w_ref[0, :]
p0 = p_ref[0, :]

num_time_steps = len(t_star) // 10
t = t_star[:num_time_steps]

for idx in range(10):
        # Get the reference solution for the current time window
        u_star = u_ref[num_time_steps * idx: num_time_steps * (idx + 1), :, :]
        print(u_ref.shape)