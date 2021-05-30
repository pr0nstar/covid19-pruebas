import numpy as np
import pandas as pd
import scipy
import scipy.integrate
import scipy.stats

import io
import csv
import json
import datetime
import os.path
import requests
import itertools as it
import unicodedata
import shutil

import matplotlib

from matplotlib import pyplot
from matplotlib import cm
from matplotlib import dates

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Hackish :S
from types import MethodType

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import ExponentialSmoothing

from github import Github
from bs4 import BeautifulSoup
from zipfile import ZipFile

np.warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ConvergenceWarning)

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

def fast_smoothing(arr, smoothing_level=.2, window_size=0):
    mask = (arr == 0)
    arr[mask] = 1
    fit = ExponentialSmoothing(
        arr, trend='mul', seasonal=None, damped=True
    ).fit(
        use_basinhopping=True, smoothing_level=smoothing_level
    )
    arr = fit.fittedvalues
    arr[mask] = 0

    if window_size:
        return moving_average(arr, window_size)

    return arr

def estimate_rt(
    new_cases,
    periodo_incubacion=5.2,
    infectivity_profile=None,
    window_size=0,
    smooth_seasons=.4
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

    no_cases_mask = new_cases < 1
    new_cases[no_cases_mask] = 1
    if smooth_seasons > 0:
        fit = ExponentialSmoothing(
            new_cases,
#             seasonal_periods=7,
            trend='mul', seasonal=None, damped=True
        ).fit(
            use_basinhopping=True,
            smoothing_level=smooth_seasons
        )
        new_cases = fit.fittedvalues

    new_cases[no_cases_mask] = 0

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
    if not 'ax' in kwargs:
    	fig, ax = pyplot.subplots()
    else:
    	ax = kwargs['ax']
    	fig = ax.get_figure()

    	del kwargs['ax']

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
    if y is None:
        y = x
        x = range(len(y))

    fig, ax = pyplot.subplots()
    ax.stem(x, y, label=label)
    ax.plot(x, moving_average(y, window_size), color='darkblue')

    pyplot.grid(b=True, color='DarkTurquoise', alpha=0.2, linestyle=':', linewidth=2)
    if label:
        ax.legend(loc='upper left')

    return ax

def plot_increments(x, y=None, label='', nth = 5, max = 10):
    if y is None:
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

def remote_path(path):
    git = Github()
    repo = git.get_repo('pr0nstar/covid19-data')

    base_path = os.path.dirname(path)
    base_name = os.path.basename(path)

    base_dir = repo.get_contents(base_path)
    file_obj = {
        os.path.basename(_.path): _ for _ in base_dir
    }[base_name]

    return file_obj.download_url

def load_testing_data():
    testing_data = pd.read_csv(
        remote_path('processed/bolivia/testing.csv')
    )
    testing_data_columns = [
        idx.lower() for idx in testing_data.columns[1:] if 'Unnamed' not in idx
    ]
    testing_data_idx = pd.MultiIndex.from_product([
        testing_data_columns, testing_data.iloc[0].reset_index()[1:][0].unique()
    ])

    testing_data['Fecha'] = pd.to_datetime(testing_data['Fecha'])
    testing_data = testing_data.set_index('Fecha')

    testing_data = testing_data.iloc[1:]
    testing_data.columns = testing_data_idx

    testing_data = testing_data.astype(np.float32)
    testing_data = testing_data.interpolate(method='linear', limit_area='inside')
    testing_data = testing_data.swaplevel(axis=1).sort_index(level=0, axis=1)

    return (
        testing_data['Sospechosos'][testing_data_columns],
        testing_data['Descartados'][testing_data_columns]
    )

CASES_DATA_NAME = {
    'confirmed': 'confirmados',
    'deaths': 'decesos'
}
ADM1_NAME = {
    'BO-B': 'beni',
    'BO-C': 'cochabamba',
    'BO-H': 'chuquisaca',
    'BO-L': 'la paz',
    'BO-N': 'pando',
    'BO-O': 'oruro',
    'BO-P': 'potosi',
    'BO-S': 'santa cruz',
    'BO-T': 'tarija'
}
COLUMNS_ORDER = [
    'la paz', 'cochabamba', 'santa cruz', 'oruro', 'potosi',
    'tarija', 'chuquisaca', 'beni', 'pando'
]
def load_data():
    data_df = pd.DataFrame([])

    for file in ['confirmed', 'deaths']:
        file_df = pd.read_csv(
            remote_path('raw/paho/{}.timeline.csv'.format(file)),
            index_col=[0],
            header=[0, 1]
        )

        try:
            file_patch_df = pd.read_csv(
                remote_path('raw/paho/{}.timeline.daily.patch.csv'.format(file)),
                index_col=[0],
                header=[0, 1]
            )

            file_df.update(file_patch_df)
        except pd.io.parsers.EmptyDataError:
            pass

        file_df.columns.names = ['', '']
        file_df.index.name = ''

        file_df = file_df['BOL']

        file_df = file_df.rename(ADM1_NAME, axis=1)
        file_df.index = pd.to_datetime(file_df.index)
        file_df.index = file_df.index - pd.Timedelta(days=1)

        file_df = file_df[COLUMNS_ORDER]
        file_df.columns = pd.MultiIndex.from_product([
            [CASES_DATA_NAME[file]], file_df.columns
        ])

        # Errores en los datos
        file_df = file_df.astype(np.float64)
        file_df = file_df.drop_duplicates().asfreq('D')
        file_df = file_df.interpolate('from_derivatives', limit_area='inside')

        file_df[(file_df.diff() < 0).shift(-1).fillna(False)] = np.nan
        file_df = file_df.interpolate('from_derivatives', limit_area='inside')

        file_df[file_df.diff() < 0] = np.nan
        file_df = file_df.interpolate('from_derivatives', limit_area='inside')

        file_df = file_df.round().dropna(how='all')
        data_df = pd.concat([data_df, file_df], axis=1)

    data_df = data_df.sort_index()
    data_df = data_df.fillna(method='ffill')

    # Aqui se cambia la definicion de caso recuperado a todos los casos 14 dias
    # despues de ser diagnosticados (deberian ser 10?)
    active_cases = data_df['confirmados'].diff().rolling(window=14).sum()
    active_cases = active_cases.fillna(data_df['confirmados'])
    active_cases.columns = pd.MultiIndex.from_product([
        ['activos'], active_cases.columns
    ])
    data_df = pd.concat([data_df, active_cases], axis=1)

    recovered_cases = data_df['confirmados'].shift(periods=14)
    recovered_cases = recovered_cases - data_df['decesos']
    recovered_cases[recovered_cases < 0] = 0
    recovered_cases.columns = pd.MultiIndex.from_product([
        ['recuperados'], recovered_cases.columns
    ])
    data_df = pd.concat([data_df, recovered_cases], axis=1)

    # Testing
    pending, discarded = load_testing_data()

    pending.columns = pd.MultiIndex.from_product([
        ['sospechosos'], pending.columns
    ])
    data_df = pd.concat([data_df, pending], axis=1)

    discarded.columns = pd.MultiIndex.from_product([
        ['descartados'], discarded.columns
    ])
    data_df = pd.concat([data_df, discarded], axis=1)

    data_df = data_df.rename({
        'confirmados': 'cases',
        'decesos': 'death',
        'activos': 'active_cases',
        'recuperados': 'recovered',
        'sospechosos': 'pending',
        'descartados': 'discarded'
    }, axis=1)

    data_df = data_df.loc[:data_df['cases'].last_valid_index()]

    return data_df

# https://www.ine.gob.bo/index.php/censos-y-proyecciones-de-poblacion-sociales/
def load_population_data(resolution, group_by=3):
    csv_file = csv.reader(open('./data/bolivia.population.final.csv'))
    data = filter(lambda _: _[group_by], csv_file)

    population_data = {}
    grouper = lambda _: (
        _[resolution].lower() if resolution == 0 else (
            _[resolution - 1].lower(), _[resolution].lower()
        )
    )
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

def open_mobility_file():
    TEMPORAL_FILE = '/tmp/movement-data.csv'
    MOVEMENT_BASE_URL = 'https://data.humdata.org'
    MOVEMENT_URL = MOVEMENT_BASE_URL + '/dataset/movement-range-maps'

    if not os.path.exists(TEMPORAL_FILE):
        cdata = requests.get(MOVEMENT_URL)
        cdata = BeautifulSoup(cdata.text)

        links = cdata.findChild('div', {'id': 'data-resources-0'}).find_all('a')
        data_link = next(
            _ for _ in links if 'href' in _.attrs and _.attrs['href'].endswith('zip')
        )
        data_link = data_link.attrs['href']
        data_link = MOVEMENT_BASE_URL + data_link if data_link.startswith('/') else data_link

        data_container = requests.get(data_link, stream=True)
        data_container = ZipFile(io.BytesIO(data_container.content))

        data_file = next(
            _ for _ in data_container.filelist if _.filename.startswith('movement-range')
        )
        data_file = data_container.open(data_file.filename)

        with open(TEMPORAL_FILE, 'wb') as disk_file:
            shutil.copyfileobj(data_file, disk_file)

        data_file.close()

    return open(TEMPORAL_FILE)

def load_mobility_data(resolution = 0, country='BOL'):
    csv_file = open_mobility_file()
    csv_file = csv.reader(csv_file, delimiter='\t')

    head = next(csv_file)
    data = [row for row in csv_file if row[1] == country]

    mobility_data = pd.DataFrame(data, columns=head)
    mobility_data['ds'] = pd.to_datetime(mobility_data['ds'])
    mobility_data = mobility_data.set_index(['polygon_id', 'ds'])

    mobility_data = mobility_data[[
        'all_day_bing_tiles_visited_relative_change',
        'all_day_ratio_single_tile_users'
    ]]
    mobility_data = mobility_data.astype(np.float64)

    vtc = pd.DataFrame()
    stc = pd.DataFrame()
    population_data = load_population_data(resolution)

    for group_key, group in population_data.items():
        group = pd.DataFrame.from_dict(group).drop('total', axis=1)
        group = group.set_index('zones')

        group_data = mobility_data.loc[group.index]

        visited_tiles_change = group_data['all_day_bing_tiles_visited_relative_change']
        visited_tiles_change = visited_tiles_change.unstack(level='ds')
        visited_tiles_change = (visited_tiles_change.T * group['weights']).sum(axis=1)

        vtc[group_key] = visited_tiles_change

        single_tile_ratio = group_data['all_day_ratio_single_tile_users']
        single_tile_ratio = single_tile_ratio.unstack(level='ds')
        single_tile_ratio = (single_tile_ratio.T * group['weights']).sum(axis=1)

        stc[group_key] = single_tile_ratio

    if len(vtc.columns) > 1:
        vtc.columns = pd.MultiIndex.from_tuples(vtc.columns)
        stc.columns = pd.MultiIndex.from_tuples(stc.columns)

    return vtc, stc

def lazy_load_data():
    data = load_data()
    data = data.groupby(level=0, axis=1).sum()

    aggregated_mobility = load_mobility_data()

    vtc, stc = load_mobility_data(1)
    key = vtc.columns.get_level_values(0)[0]
    local_mobility = vtc[key], stc[key]

    return data, aggregated_mobility, local_mobility
