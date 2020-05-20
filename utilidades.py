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

        model_data = [_[-2] for _ in solution['y']]
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

def moving_average(arr, by):
    return np.convolve(
        np.pad(arr, (0, 1), 'edge'),
        np.ones((by,)) / by,
        mode='same'
    )[:-1]

def estimate_rt(
    new_cases,
    periodo_incubacion=5.2,
    infectivity_profile=None,
    window_size=0
):
    '''
    https://stochastik-tu-ilmenau.github.io/COVID-19/reports/repronum/repronum.pdf
    '''
    if infectivity_profile is None:
        infectivity_profile = np.append(np.arange(0, 4) / 3., [
            1., 1., *np.arange(5, -1, -1) / 5.
        ])
        infectivity_profile = infectivity_profile / sum(infectivity_profile)

    res = []

    for idx in range(int(periodo_incubacion) + 1, len(new_cases) - 1):
        cases = new_cases[:idx]
        difference = len(cases) - len(infectivity_profile)

        if difference > 0:
            profile = np.pad(
                infectivity_profile,
                (difference, 0),
                'constant',
                constant_values=(0,)
            )

        else:
            profile = infectivity_profile[-1 * len(cases):]

        tau = max(1., np.sum(cases * profile))
        res.append((
            new_cases[idx + 1] / tau,
            new_cases[idx + 1] / tau ** 2
        ))

    r_ts, r_ts_err = zip(*res)

    r_ts = np.pad(
        r_ts[len(infectivity_profile):],
        (len(infectivity_profile), 0),
        'constant',
        constant_values=(0,)
    )
    r_ts_err = np.pad(
        r_ts_err[len(infectivity_profile):],
        (len(infectivity_profile), 0),
        'constant',
        constant_values=(0,)
    )
    r_ts[r_ts < 0.] = 0.01
    r_ts[r_ts > 6.] = 6.
    r_ts_err[r_ts_err < 0.] = 0.01
    r_ts_err[r_ts_err > 6.] = 6.

    r_ts_err = np.sqrt(r_ts_err)

    qnorm = scipy.stats.norm.ppf(1 - 0.05 / 2)

    if window_size > 0:
        r_ts = moving_average(r_ts, window_size)
        r_ts_err = moving_average(r_ts_err, window_size)

    return r_ts, r_ts - qnorm * r_ts_err, r_ts + qnorm * r_ts_err

pyplot.rcParams['figure.figsize'] = [40/2.54, 20/2.54]
def plot(x, *curves, **kwargs):
    fig, ax = pyplot.subplots()
    for idx, curve in enumerate(curves):
        label = kwargs['labels'][idx] if 'labels' in kwargs else None
        ax.plot(x[:len(curve)], curve, label=label)

    pyplot.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)

    if 'labels' in kwargs:
        ax.legend(loc='upper left')

    return ax

def plot_moving_averaged(x, y = None, label = None):
    if not y:
        y = x
        x = range(len(y))

    fig, ax = pyplot.subplots()
    ax.stem(x, y, label=label)
    ax.plot(moving_average(y, 4))

    pyplot.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)
    if label:
        ax.legend(loc='upper left')

    return ax

def draw_effective_rts(
    new_cases,
    window_size = 0,
    infectivity_profile = None,
    ax = None
):
    if ax is None:
        fig, ax = pyplot.subplots()

    ax.stem([0] * 2 + list(new_cases), label='Casos Diarios')
    r_ts, r_ts_min, r_ts_max = estimate_rt(
        new_cases,
        window_size = window_size,
        infectivity_profile = infectivity_profile
    )

    ax2 = ax.twinx()
    _ = ax2.plot(r_ts, color='green', label='R_t proyectado')

    ax2.fill_between(
        range(len(r_ts)),
        r_ts_min,
        r_ts_max,
        alpha=0.2
    )

    ax2.axhline(1.)
    ax2.legend(loc='upper left')
    ax2.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)

    return ax, ax2

def get_inflections(curve):
    # max/min
    d = np.diff(curve)
    # inflections
    dd = np.diff(d)

    # donde haya un cambio de signo
    mmp = np.where(np.diff(np.sign(d)))[0]
    inp = np.where(np.diff(np.sign(dd)))[0]

    # no consecutivo
    return (
        mmp[np.diff(mmp, prepend=[-1]) != 1],
        inp[np.diff(inp, prepend=[-1]) != 1]
    )

def draw_curve_inflection(
    x, y = None,
    label = None,
    ax = None,
    direction = 'h',
    **kwargs
):
    if ax is None:
        fig, ax = pyplot.subplots()

    if y is None:
        curve = np.array(x)
        x = range(len(curve))
    else:
        curve = np.array(y)

    ax.plot(x, curve, label=label, **kwargs)

    point_groups = get_inflections(curve)
    for idx, point_group in enumerate(point_groups):
        for point in point_group:
            x_point = point + idx + 1

            ax.plot(x[x_point], curve[x_point], marker='D')

            if direction == 'h':
                ax.axhline(curve[x_point], linestyle='dotted', color='black', alpha=0.6)
            elif direction == 'v':
                ax.axvline(x[x_point], linestyle='dotted', color='black', alpha=0.6)

    if label:
        ax.legend(loc='upper left')

    ax.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)

    return ax

def describe_simulation(
    solution_dict,
    params_dict = None,
    days = None,
    infectivity_profile = None
):
    params_dict = params_dict if params_dict is not None else {}
    fig, axs = pyplot.subplots(
        figsize=(50 / 2.54, 2 * 25 / 2.54),
        ncols=2,
        nrows=3 + int(np.floor(len([
            _ for _ in params_dict.values() if type(_) == list
        ]) / 2.))
    )

    if 'population' not in params_dict:
        population = solution_dict['susceptible'][0]
    else:
        population = params_dict['population']

    draw_curve_inflection(
        100. * solution_dict['susceptible'] / population,
        label = 'Susceptibles (%)',
        ax = axs[0][0]
    )
    ax = draw_curve_inflection(
        100. * solution_dict['infected'] / population,
        label='Infectados (%)',
        ax = axs[0][1],
        direction = 'v'
    )
    ax.plot(100. * solution_dict['exposed'] / population, label='Expuestos (%)')
    ax.fill_between(
        range(len(solution_dict['infected'])),
        [0] * len(solution_dict['infected']),
        100. * (solution_dict['infected'] + solution_dict['exposed']) / population,
        alpha=0.1,
        label='E + I'
    )
    ax.legend(loc='upper left')

    draw_curve_inflection(
        100. * solution_dict['recovered'] / population,
        label='Recuperados (%)',
        ax = axs[1][0]
    )
    draw_curve_inflection(
        100. * solution_dict['death'] / population,
        label='Fallecidos (%)',
        ax = axs[1][1]
    )

    new_cases = np.diff(
        solution_dict['infected'] + solution_dict['exposed'],
        prepend=[solution_dict['infected'][0]]
    )
    _, ax = draw_effective_rts(
        new_cases,
        ax=axs[2][0],
        window_size=5,
        infectivity_profile=infectivity_profile
    )
    if 'R0' in params_dict:
        delay = len(infectivity_profile) if infectivity_profile is not None else 0
        ax.plot(
            (params_dict['R0'] * (solution_dict['susceptible'] / population))[delay:],
            label='R_t efectivo'
        )
        ax.legend(loc='upper left')

    for idx, key in enumerate(params_dict.keys()):
        params = params_dict[key]

        if type(params) != list:
            continue

        curve = np.hstack([days[_] * [params[_]] for _ in range(len(days))])

        draw_curve_inflection(
            curve,
            label=key,
            ax = axs[2 + int((idx + 1) / 2)][int((idx + 1) % 2)]
        )

    return axs

# https://github.com/mauforonda
def load_data():
    final_data = []

    with open('./data/covid19-bolivia2/nacional.csv') as f:
        csv_file = csv.reader(f)
        data = [
            [
                int(_.replace('.', '')) for _ in line[1:]
            ] for idx, line in enumerate(csv_file) if idx > 0
        ]
        active_cases = np.array([_[0] - sum(_[1:]) for _ in data])
        data = [np.array(_) for _ in zip(*data)]
        data.insert(0, active_cases)

        final_data.extend(data)

    with open('./data/covid19-bolivia/descartados.csv') as f:
        csv_file = csv.reader(f)
        data = [
            [
                int(_.replace('.', '')) for _ in line[1:]
            ] for idx, line in enumerate(csv_file) if idx > 0
        ]
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

# https://github.com/CSSEGISandData/COVID-19
def load_data_jhu(country):
    FILES = [
        'time_series_covid19_confirmed_global.csv',
        'time_series_covid19_deaths_global.csv',
        'time_series_covid19_recovered_global.csv'
    ]
    PATH = './data/COVID-19/csse_covid_19_data/csse_covid_19_time_series/{}'
    country = country.lower()
    final_data = []

    for file_name in FILES:
        with open(PATH.format(file_name)) as f:
            csv_file = csv.reader(f)
            data = np.array([
                line[4:] for idx, line in enumerate(csv_file) if (
                    idx > 0 and line[1].lower() == country
                )
            ]).astype(np.float)
            data = data.sum(axis=0)
            final_data.append(data)

    start = np.argwhere(final_data[0])[0][0]
    for idx, data in enumerate(final_data):
        final_data[idx] = data[start:]

    cases, deaths, recovered = final_data

    active_cases = np.array([
        cases[_] - deaths[_] - recovered[_] for _ in range(len(cases))
    ])
    final_data.insert(0, active_cases)

    return np.array(final_data)
