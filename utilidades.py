import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import scipy.stats

import csv
import json
import datetime
import os.path
import urllib.request
import itertools as it
import unicodedata

import matplotlib
from matplotlib import pyplot
from matplotlib import cm

# Hackish :S
from types import MethodType


np.warnings.filterwarnings('ignore')

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

    new_cases = new_cases.tolist()
    new_cases.append(0)
    new_cases = np.array(new_cases)

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

    r_ts_max = r_ts + qnorm * r_ts_err
    r_ts_min = r_ts - qnorm * r_ts_err
    r_ts_min[r_ts_min < 0] = 0

    return r_ts, r_ts_min, r_ts_max

pyplot.rcParams['figure.figsize'] = [50/2.54, 25/2.54]
def plot(x, *curves, **kwargs):
    fig, ax = pyplot.subplots()

    labels = []
    if 'labels' in kwargs:
        labels = kwargs['labels']
        del kwargs['labels']

    for idx, curve in enumerate(curves):
        label = labels[idx] if labels else None
        ax.plot(x[:len(curve)], curve, label=label, **kwargs)

    pyplot.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)

    if labels:
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

def plot_increments(x, y=None, label='', nth = 5, max = 10):
    if not y:
        y = x
        x = range(len(y))

    fig, ax = pyplot.subplots()

    cum_cases = accumulate(y, nth)
    cum_cases[cum_cases < 1.] = 1.

    inc_cases = cum_cases[1:] / cum_cases[:-1]
    inc_cases[inc_cases > 10.] = 10.
    inc_cases = np.pad(inc_cases, (1, 0), 'constant', constant_values=(0,))
    inc_cases = np.repeat(inc_cases, nth)[-1 * len(x):]

    ax.step(x, inc_cases, label=label)

    ax.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)
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

def do_read_csv(file_name):
    with open(file_name) as f:
        return list(csv.reader(f))

def do_process_label(label):
    return unicodedata.normalize(
        'NFD', label
    ).encode(
        'ascii', 'ignore'
    ).decode("utf-8").lower()

FIRST_DAY = '2020-03-21'
def get_today(data):
    date = datetime.datetime.strptime(
        FIRST_DAY, '%Y-%m-%d'
    ) + datetime.timedelta(days=len(data[0]) - 1)
    return '{}'.format(date.strftime('%Y-%m-%d'))

def download_testing_data_old(write_to):
    URL = 'https://app.flourish.studio/api/data_table/3987481/csv'

    response = urllib.request.urlopen(URL)
    data = response.read().decode('utf-8')

    with open(write_to, 'w') as f:
        f.write(data)

def download_testing_data(write_to):
    URL = 'https://flo.uri.sh/visualisation/2519845/embed'

    req = urllib.request.Request(
        URL, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'}
    )
    response = urllib.request.urlopen(req)
    data = response.read().decode('utf-8')

    store_to = []
    look_for = [
        '_Flourish_data_column_names = ',
        '_Flourish_data = '
    ]

    for line in data.split('\n')[-10:]:
        for wildcard in look_for:
            if not wildcard in line:
                continue

            _, line_data = line.split(wildcard)
            store_to.append(line_data[:-1])

    if not store_to:
        raise Exception('No funca wey')

    header, remote_data = [json.loads(_)['data'] for _ in store_to]
    remote_data = [header] + remote_data
    local_data = [([row['label']] + row['value']) for row in remote_data]

    with open(write_to, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(local_data)

# https://muywaso.com/especial-de-datos-muy-waso-sobre-el-coronavirus-en-bolivia/
def load_testing_data(aggregate = False):
    PATH = './data/testing.muywaso.csv'
    if not os.path.isfile(PATH):
        download_testing_data(PATH)

    response = do_read_csv(PATH)
    response_header = [do_process_label(_) for _ in response[0][1:]]

    response = np.array([[
        _ for _ in row[1:] if len(_)
    ] for row in response[1:]], dtype=np.int32)

    response = response.cumsum(axis=0)

    if aggregate:
        return response.sum(axis=1)
    else:
        return dict((zip(response_header, response.T)))

# https://github.com/mauforonda
def load_data(aggregate = True):
    FILES = ['confirmados', 'decesos', 'recuperados']
    response = [do_read_csv(
        './data/covid19-bolivia/{}.csv'.format(file_name)
    ) for file_name in FILES]

    response_header = [do_process_label(_) for _ in response[0][0][1:]]
    response = np.array([[
        row[1:] for row in data[1:]
    ][::-1] for data in response], dtype=np.int32)

    if (response[0][-1] == response[0][-2]).all():
        response = response[:, :-1]

    testing_data = load_testing_data(aggregate)

    if aggregate:
        response = np.array([np.sum(_, axis=1) for _ in response])
        active_cases = response.T[:,0] - response.T[:,1:].sum(axis=1)
        final_response = np.insert(response, 0, active_cases, axis=0)

        testing_data = np.pad(
            testing_data,
            (0, len(active_cases) - len(testing_data)),
            'constant',
            constant_values=(0,)
        )
        final_response = np.append(final_response, [testing_data], axis=0)

    else:
        final_response = {}
        groups = response.T

        for idx, response in enumerate(groups):
            response = response.T

            active_cases = response.T[:,0] - response.T[:,1:].sum(axis=1)
            response = np.insert(response, 0, active_cases, axis=0)

            group_testing_data = testing_data[response_header[idx]]
            group_testing_data = np.pad(
                group_testing_data,
                (0, len(active_cases) - len(group_testing_data)),
                'constant',
                constant_values=(0,)
            )

            final_response[response_header[idx]] = np.append(
                response, [group_testing_data], axis=0
            )

    return final_response

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

# https://www.ine.gob.bo/index.php/censos-y-proyecciones-de-poblacion-sociales/
def load_population_data(resolution, group_by=3):
    csv_file = csv.reader(open('./data/bolivia.population.final.csv'))
    data = filter(lambda _: _[group_by], csv_file)

    population_data = {}
    grouper = lambda _: _[resolution] if resolution == 0 else (_[resolution - 1], _[resolution])
    for group_key, group in it.groupby(data, key=grouper):
        group_data = population_data[group_key] = {
            'zones': [],
            'weights': [],
            'total': 0
        }
        for element in group:
            group_data['zones'].append(element[3])
            group_data['weights'].append(int(element[4]))
            group_data['total'] = group_data['total'] + int(element[4])

        group_data['weights'] = np.array(group_data['weights']) / group_data['total']

    return population_data

# https://data.humdata.org/dataset/movement-range-maps
def load_mobility_data(resolution = 0):
    mobility_data = {}

    csv_file = csv.reader(open('./data/facebook-mobility.bol.txt'), delimiter='\t')
    data = list(csv_file)

    for zone_key, group in it.groupby(data, key=lambda _: _[3]):
        data = []
        for element in group:
            data.append(
                (element[0], float(element[5]), float(element[6]))
            )

        if len(data) > 30:
            mobility_data[zone_key] = data

    vtc = pd.DataFrame()
    stc = pd.DataFrame()
    population_data = load_population_data(resolution)

    for group_key, group in population_data.items():
        group_data = []
        for idx in range(len(group['zones'])):
            zone_key = group['zones'][idx]

            if zone_key not in mobility_data:
                continue

            dates, visited_tiles_change, single_tile_ratio = zip(
                *mobility_data[zone_key]
            )
            weight = group['weights'][idx]

            group_data.extend(
                zip(
                    dates,
                    np.array(visited_tiles_change) * weight,
                    np.array(single_tile_ratio) * weight
                )
            )

        if not group_data:
            continue

        group_data = pd.DataFrame(group_data)
        group_data.columns = (
            'date', 'visited_tiles_change', 'single_tile_ratio'
        )

        group_data = group_data.groupby('date').sum()

        vtc[group_key] = group_data['visited_tiles_change']
        stc[group_key] = group_data['single_tile_ratio']

    if len(vtc.columns) > 1:
        vtc.columns = pd.MultiIndex.from_tuples(vtc.columns)
        stc.columns = pd.MultiIndex.from_tuples(stc.columns)

    return vtc, stc

def lazy_load_data(where = None):
    if where is None:
        data = load_data(aggregate=True)
        aggregated_mobility = load_mobility_data()

        vtc, stc = load_mobility_data(1)
        key = vtc.columns.get_level_values(0)[0]
        local_mobility = vtc[key], stc[key]

    else:
        data = load_data(aggregate=False)[where]
        vtc, stc = load_mobility_data(1)
        key = vtc.columns.get_level_values(0)[0]
        aggregated_mobility = vtc[key][where.upper()], stc[key][where.upper()]

        vtc, stc = load_mobility_data(2)
        local_mobility = vtc[where.upper()], stc[where.upper()]

    return data, aggregated_mobility, local_mobility
