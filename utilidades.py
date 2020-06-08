import numpy as np
import scipy
import scipy.integrate
import scipy.stats

import csv
import os.path
import urllib.request

import matplotlib
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

        to_time = acc_days + point_days + int(bool(point_idx))
        steps = np.arange(start=acc_days, stop=to_time, step=step)

        solution = scipy.integrate.solve_ivp(
            lambda _, model_data_t: model_fn(model_data_t, **params_t),
            y0 = model_data,
            t_eval = steps,
            t_span = (acc_days, to_time)
        )

        if not final_solution:
            final_solution = solution
            # final_solution['t'] = final_solution['t']
            # final_solution['y'] = [solution_y for solution_y in final_solution['y']]

        else:
            final_solution['t'] = np.append(final_solution['t'], solution['t'][1:])
            final_solution['y'] = [
                np.append(
                    final_solution['y'][idx], solution_y[1:]
                ) for idx, solution_y in enumerate(solution['y'])
            ]

        model_data = [_[-1] for _ in solution['y']]
        acc_days = to_time - 1

    return final_solution

def moving_average(arr, by):
    pad = int(by / 2)
    return np.convolve(
        np.pad(arr, (0, pad), 'edge'),
        np.ones((by,)) / by,
        mode='same'
    )[:-pad]

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
            *np.arange(5, -1, -1) / 5.
        ])
        infectivity_profile = infectivity_profile / sum(infectivity_profile)

    res = []

    if window_size > 0:
        new_cases = moving_average(new_cases, window_size)

    for idx in range(int(periodo_incubacion) + 1, len(new_cases)):
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
            new_cases[idx] / tau,
            new_cases[idx] / tau ** 2
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

    r_ts, r_ts_err = r_ts[:-1], r_ts_err[:-1]

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

def plot_moving_averaged(x, y = None, label = None, window_size = 4):
    if not y:
        y = x
        x = range(len(y))

    fig, ax = pyplot.subplots()
    ax.stem(x, y, label=label)
    ax.plot(moving_average(y, window_size), color='darkblue')

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

    ax = draw_curve_inflection(
        100. * (solution_dict['infected'] - solution_dict['exposed']) / population,
        label='Infecciosos (%)',
        ax = axs[0][1],
        direction = 'v'
    )
    ax.plot(100. * solution_dict['exposed'] / population, label='Expuestos (%)')
    ax.fill_between(
        range(len(solution_dict['infected'])),
        [0] * len(solution_dict['infected']),
        100. * solution_dict['infected'] / population,
        alpha=0.1,
        label='I + E'
    )
    ax.legend(loc='upper left')

    draw_curve_inflection(
        100. * solution_dict['susceptible'] / population,
        label = 'Susceptibles (%)',
        ax = axs[0][0]
    )

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

    new_cases = solution_dict['new_cases']
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

def phase_transition(solution, population_t0, ax = None):
    ax = draw_curve_inflection(
        100. * solution['susceptible'] / population_t0,
        100. * solution['infected'] / population_t0,
        direction = 'v',
        ax = ax
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title('Diagrama de fase')
    ax.set(xlabel='Susceptibles (%)', ylabel='Infectados (%)')

    return ax

# https://muywaso.com/especial-de-datos-muy-waso-sobre-el-coronavirus-en-bolivia/
def load_testing_data():
    PATH = './data/testing.muywaso.csv'
    URL = 'https://app.flourish.studio/api/data_table/3987481/csv'

    if not os.path.isfile(PATH):
        response = urllib.request.urlopen(URL)
        data = response.read().decode('utf-8')

        with open(PATH, 'w') as f:
            f.write(data)

    data = []

    with open(PATH) as f:
        csv_file = csv.reader(f)
        data = [
            [
                int(_.replace('.', '')) if _ else 0 for _ in line[1:]
            ] for idx, line in enumerate(csv_file) if idx > 0
        ]
        data = [sum(_) for _ in data]
        data = np.pad(data, (11, 0), 'constant', constant_values=(0,))

    return data

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

    data = load_testing_data()
    data = np.pad(
        data,
        (0, len(final_data[0]) - len(data)),
        'constant',
        constant_values=(0,)
    )
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
