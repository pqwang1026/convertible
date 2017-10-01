from convertible_solver.game_solver import *
import utils.mp_wrapper
from solver.fp_solver import *
import utils.simplex as splx
import utils.distribution

T = 3
model_tesla = CCBModel.get_model(
    r=0.01,
    c=0.02,
    T=T,
    e=10,
    p=1000,
    M=100e6,
    N=4e6,
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


def f(sp: splx.SimplexPoint):
    """
    Input is a simplex point, we first transform it into a probability distribution, and treat it as I to solve the mfg.
    The return is again a simplex point.
    """
    major_stopping_dist = model.get_random_initial_dist()
    # minor_stopping_dist = model.get_random_initial_dist()
    minor_stopping_dist = utils.distribution.Distribution([i * model.T / model.time_num_grids for i in range(0, model.time_num_grids + 1)], sp)

    solver = GameSolver(
        model,
        major_stopping_dist,
        minor_stopping_dist,
        num_max_iter=10,
        num_mc=1000,
        precision=0.05,
    )
    solver.update_()
    return splx.SimplexPoint(solver.minor_stopping_dist.pdf.values)


# solver = IterataiveSolver(f, n=T)
# print(solver.solve())

solver = KakutaniSolver(f, n=T)
# print(solver.solve(k=4))
solver.solve_traversal(k=4)
