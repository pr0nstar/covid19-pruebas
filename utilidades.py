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

        steps = np.arange(start=acc_days, stop=acc_days + point_days + 1, step=step)
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
            final_solution['t'] = final_solution['t'][:-1]
            final_solution['y'] = [solution_y[:-1] for solution_y in final_solution['y']]

        else:
            final_solution['t'] = np.append(final_solution['t'][:-1], solution['t'])
            final_solution['y'] = [
                np.append(
                    final_solution['y'][idx][:-1], solution_y
                ) for idx, solution_y in enumerate(solution['y'])
            ]

    return final_solution


pyplot.rcParams['figure.figsize'] = [50/2.54, 22/2.54]
def plot(x, *curves, **kwargs):
    fig, ax = pyplot.subplots()
    for idx, curve in enumerate(curves):
        label = kwargs['labels'][idx] if 'labels' in kwargs else None
        ax.plot(x[:len(curve)], curve, label=label)

    pyplot.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)

    if 'labels' in kwargs:
        ax.legend(loc='upper left')

    return ax

# https://github.com/mauforonda
def load_data():
    final_data = []

    with open('./data/covid19-bolivia2/nacional.csv') as f:
        csv_file = csv.reader(f)
        data = [[int(_) for _ in line[1:]] for idx, line in enumerate(csv_file) if idx > 0]

        final_data.extend(zip(*data))

    with open('./data/covid19-bolivia/descartados.csv') as f:
        csv_file = csv.reader(f)
        data = [[int(_) for _ in line[1:]] for idx, line in enumerate(csv_file) if idx > 0]
        data = [sum(_) for _ in data]
        data = np.pad(data, (0, 12), 'constant', constant_values=(0,))
        data = np.array([data[_] - data[_ + 1] for _ in range(len(data) - 1)], dtype='float64')

        # No hay una forma mas facil?
        zeros = np.flatnonzero(data == 0)
        zeros_counts = np.diff(np.flatnonzero(np.concatenate(
            ([True], (zeros[1:] - 1) != zeros[:-1], [True])
        )))

        acc = 0
        for zero_count in zeros_counts:
            pos = zeros[acc]
            if pos > 0:
                data[pos - 1:pos + zero_count] = data[pos - 1] / (zero_count + 1.)
            acc = acc + zero_count

        data = np.flip(data) + np.diff(final_data[1], prepend=[final_data[1][0]])
        final_data.append(data)

    return np.array(final_data)
