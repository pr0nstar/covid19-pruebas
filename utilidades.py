import csv
import numpy as np
import scipy
import scipy.integrate

from matplotlib import pyplot


def run(model_fn, model_data, days, acc_days=0,  step=0.1, **params):
    if type(days) is not list:
        days = [days]

    final_solution = None

    for point_idx, point_days in enumerate(days):
        params_t = params.copy()
        for key in params_t:
            if type(params_t[key]) == list:
                params_t[key] = params_t[key][point_idx]

        steps = np.arange(start=acc_days, stop=acc_days + point_days, step=step)
        solution = scipy.integrate.solve_ivp(
            lambda _, model_data_t: model_fn(model_data_t, **params_t),
            y0 = model_data,
            t_eval = steps,
            t_span = (acc_days, acc_days + point_days)
        )

        model_data = [_[-1] for _ in solution['y']]
        acc_days = acc_days + point_days

        if not final_solution:
            final_solution = solution
        else:
            final_solution['t'] = np.append(final_solution['t'], solution['t'])
            final_solution['y'] = [
                np.append(final_solution['y'][idx], solution_y) for idx, solution_y in enumerate(solution['y'])
            ]

    return final_solution


pyplot.rcParams['figure.figsize'] = [50/2.54, 22/2.54]
def plot(x, *curves):
    fig, ax = pyplot.subplots()
    for curve in curves:
        ax.plot(x[:len(curve)], curve)

    pyplot.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)

    return ax

# https://github.com/mauforonda
def load_data(path='./data/covid19-bolivia2/nacional.csv'):
    with open(path) as f:
        csv_file = csv.reader(f)
        data = [[int(_) for _ in line[1:]] for idx, line in enumerate(csv_file) if idx > 0]
        data = zip(*data)

    return list(data)
