from convertible_solver.game_solver import *
import utils.mp_wrapper

T = 3
model_tesla = CCBModel.get_model(
    r=0.01,
    c=0.02,
    T=T,
    e=10,
    p=1000,
    M=100e6,
    N=1e6,
    k=1,
    d=20,
    nu=0.01,
    sigma=0.8,
    sigma_R=0.02,
    R_lt=0.015,
    theta=0.1,
    v_0=100,
    R_0=0.02,
    lambd=1,
    time_num_grids=T,
    state_num_grids=50,
)

# set up the game model

model = model_tesla


# solve the model

# major_stopping_dist = model.get_initial_major_stopping_dist()
# minor_stopping_dist = model.get_initial_minor_stopping_dist()
def try_():
    major_stopping_dist = model.get_random_initial_dist()
    minor_stopping_dist = model.get_random_initial_dist()

    solver = GameSolver(
        model,
        major_stopping_dist,
        minor_stopping_dist,
        num_max_iter=25,
        num_mc=1000,
        precision=0.05,
    )
    solver.solve()
    plt.show()


if __name__ == '__main__':
    try_()
    # utils.mp_wrapper.run(try_, args=[{} for i in range(0, 6)])
