import pathlib
import pandas as pd
import matplotlib
import warnings
import tensorflow as tf
import datetime

from tqdm import tqdm_notebook as tqdm
from numpy import array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import LSTM
from dateutil.relativedelta import relativedelta

matplotlib.use('nbagg')
warnings.filterwarnings("ignore")


def get_base_date(train_path: str, submission_path: str, test_path: str):
    train = pd.read_csv(train_path)
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)
    return train, submission, test


def clean_data(data_to_clean: pd.DataFrame) -> pd.DataFrame:
    return data_to_clean.fillna('NoState')


def get_newly_added(world_data_):
    world_data_ = world_data_.sort_values(['Country/Region', 'ObservationDate'])
    temp = [0 * i for i in range(len(world_data_))]
    world_data_['New Confirmed'] = temp
    world_data_['New Death'] = temp
    for i in tqdm(range(1, len(world_data_))):
        if world_data_['Country/Region'].iloc[i] == world_data_['Country/Region'].iloc[i - 1]:
            if world_data_['Deaths'].iloc[i] < world_data_['Deaths'].iloc[i - 1]:
                world_data_['Deaths'].iloc[i] = world_data_['Deaths'].iloc[i - 1]
            if world_data_['Confirmed'].iloc[i] < world_data_['Confirmed'].iloc[i - 1]:
                world_data_['Confirmed'].iloc[i] = world_data_['Confirmed'].iloc[i - 1]
            world_data_['New Confirmed'].iloc[i] = world_data_['Confirmed'].iloc[i] - world_data_['Confirmed'].iloc[
                i - 1]
            world_data_['New Death'].iloc[i] = world_data_['Deaths'].iloc[i] - world_data_['Deaths'].iloc[i - 1]
        else:
            world_data_['New Confirmed'].iloc[i] = world_data_['Confirmed'].iloc[i]
            world_data_['New Death'].iloc[i] = world_data_['Deaths'].iloc[i]
    return world_data_


def create_train_dataset(target, n_steps, train, pivot_date, unique_regions, states_per_regions):
    train = train.query("ObservationDate<" + pivot_date)
    x = []
    y = []
    for k in tqdm(range(len(unique_regions))):
        for state in states_per_regions[k]:
            temp = train[(train['Country/Region'] == unique_regions[k]) & (train['Province/State'] == state)]
            sequence = list(temp[target])
            for i in range(len(sequence)):
                end_ix = i + n_steps
                if end_ix > len(sequence) - 1:
                    break
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
                if seq_y != 0:
                    x.append(seq_x)
                    y.append(seq_y)
    return array(x), array(y)


def create_test_dataset(target, n_steps, train, pivot_date, unique_regions, states_per_regions):
    train = train.query("ObservationDate<" + pivot_date)
    x = []
    regs = []
    for k in tqdm(range(len(unique_regions))):
        for state in states_per_regions[k]:
            temp = train[(train['Country/Region'] == unique_regions[k]) & (train['Province/State'] == state)]
            sequence = temp[target]
            x.append(sequence[len(sequence) - n_steps:len(sequence) + 1])
            regs.append((unique_regions[k], state))
    return array(x), regs


def pred(model, data):
    y_pred = model.predict(data)
    return y_pred


def forcast(model, data, start_date, num_days, steps, regs):
    """
    Prediction function
    Training model with confirmed cases and deaths.
    start_date: Prediction start_date
    num_days: Amount of days to perform the forecast
    """
    res_ = dict()
    for i in range(len(data)):
        res_[i] = []
    y_pred = pred(model, data)
    dates = []
    date1 = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    for j in range(1, num_days + 1):
        for i in range(len(data)):
            cur_window = list(data[i][0][1:steps + 1])
            res_[i].append(cur_window[-1])
            cur_window.append(y_pred[i])
            data[i][0] = cur_window
        y_pred = pred(model, data)
        dates.append(date1.strftime("%Y-%m-%d"))
        date1 += relativedelta(days=1)
    res = pd.DataFrame(pd.DataFrame(pd.DataFrame(res_).values.T))
    res.columns = dates
    res['Country/State'] = regs
    print(res)
    return res


def prepare_submission(res_c, res_d, res_nc, test, pivot_date):
    test = test.query("Date>=" + pivot_date)
    index = dict()
    for i in range(len(res_c)):
        index[res_c.iloc[i]['Country/State']] = i
    pred_c = []
    pred_d = []
    pred_nc = []
    for i in tqdm(range(len(test))):
        if (test.iloc[i]['Country_Region'], test.iloc[i]['Province_State']) in index:
            loc = index[(test.iloc[i]['Country_Region'], test.iloc[i]['Province_State'])]
            pred_c.append(res_c.iloc[loc][test.iloc[i]['Date']])
            pred_d.append(res_d.iloc[loc][test.iloc[i]['Date']])
            pred_nc.append(res_nc.iloc[loc][test.iloc[i]['Date']])
    test['ConfirmedCases'] = pred_c
    test['Fatalities'] = pred_d
    test['New Confirmed'] = pred_nc
    res_regional = test
    res = test.drop(columns=['Province_State', 'Country_Region', 'Date', 'New Confirmed'])
    return res, res_regional


def get_countrywise_forcast_(covid_timeseries, pivot_date, target, country_name, state_name, num_days, res_regional):
    """
    Function for countries forecast
    """
    temp = covid_timeseries[(covid_timeseries['Country/Region'] == country_name) & (
            covid_timeseries['Province/State'] == state_name)].query("ObservationDate>=" + pivot_date)
    x_truth = temp.ObservationDate
    y_truth = temp[target]
    pred_ = res_regional[
        (res_regional['Country_Region'] == country_name) & (res_regional['Province_State'] == state_name)]

    x_pred = pred_.Date[0:num_days]
    y_pred = pred_[target][0:num_days]

    return list(x_truth), list(y_truth), list(x_pred), list(y_pred)


def get_countrywise_forcast(data, pivot_date, country_name, state_name, num_days, res_regional):
    """
    Function to predict confirmed cases by country
    """
    temp = data[(data['Country/Region'] == country_name) & (data['Province/State'] == state_name)].query(
        "ObservationDate>=" + pivot_date)

    x_truth = temp.ObservationDate
    y_truth = temp.Confirmed
    pred_ = res_regional[
        (res_regional['Country_Region'] == country_name) & (res_regional['Province_State'] == state_name)]

    x_pred = pred_.Date[0:num_days]
    y_pred = pred_.ConfirmedCases[0:num_days]
    return list(x_truth), list(y_truth), list(x_pred), list(y_pred)


def get_countrywise_forcast_deaths(data, pivot_date, country_name, state_name, num_days, res_regional):
    """
    Function to predict death cases by country
    """
    temp = data[(data['Country/Region'] == country_name) & (data['Province/State'] == state_name)].query(
        "ObservationDate>=" + pivot_date)
    x_truth = temp.ObservationDate
    y_truth = temp.Deaths
    pred_ = res_regional[
        (res_regional['Country_Region'] == country_name) & (res_regional['Province_State'] == state_name)]
    x_pred = pred_.Date[0:num_days]
    y_pred = pred_.Fatalities[0:num_days]
    return list(x_truth), list(y_truth), list(x_pred), list(y_pred)


def get_countries_prediction(countries: list[dict]):
    current_dir = pathlib.Path().resolve()
    train_data_path = f'{current_dir}/app/data/train.csv'
    submission_data_path = f'{current_dir}/app/data/submission.csv'
    test_data_path = f'{current_dir}/app/data/test.csv'

    train, submission, test = get_base_date(train_data_path, submission_data_path, test_data_path)

    train = clean_data(train)
    test = clean_data(test)

    custom_columns = {
        'ConfirmedCases': 'Confirmed',
        'Fatalities': 'Deaths',
        'Country_Region': 'Country/Region',
        'Province_State': 'Province/State',
        'Date': 'ObservationDate'
    }
    train = train.rename(columns=custom_columns)
    numeric_columns = ['Confirmed', 'Deaths']
    for column in numeric_columns:
        temp = [int(i) for i in train[column]]
        train[column] = temp

    # Get regions and countries list
    unique_regions = train['Country/Region'].unique()
    states_per_regions = []
    for reg in tqdm(unique_regions):
        states_per_regions.append(train[train['Country/Region'] == reg]['Province/State'].unique())

    columns_to_group = ['ObservationDate', 'Country/Region', 'Province/State']

    # Temp data for countries with new number of daily cases
    covid_timeseries = train.groupby(columns_to_group)['Confirmed', 'Deaths'].sum()
    covid_timeseries = covid_timeseries.reset_index().sort_values('ObservationDate')
    covid_timeseries = get_newly_added(covid_timeseries)

    # Data for training
    # pivot_date -> previous date data for training
    # forcast_start_date -> date from prediction will be done.
    n_steps = 7
    pivot_date = "'2020-04-02'"
    forcast_start_date = '2020-04-02'
    print('Preparación de conjuntos de datos de casos confirmados acumulados')
    X_c, y_c = create_train_dataset('Confirmed', n_steps, train, pivot_date, unique_regions, states_per_regions)
    print('Preparación de conjuntos de datos de nuevos confirmados..')
    X_nc, y_nc = create_train_dataset('New Confirmed', n_steps, covid_timeseries, pivot_date, unique_regions,
                                      states_per_regions)
    test_confirmed, regs = create_test_dataset('Confirmed', n_steps, train, pivot_date, unique_regions,
                                               states_per_regions)
    test_nc, reg_nc = create_test_dataset('New Confirmed', n_steps, covid_timeseries, pivot_date, unique_regions,
                                          states_per_regions)
    print('Preparacion de conjunto de datos de muertos')
    X_d, y_d = create_train_dataset('Deaths', n_steps, train, pivot_date, unique_regions, states_per_regions)
    test_deaths, regs = create_test_dataset('Deaths', n_steps, train, pivot_date, unique_regions, states_per_regions)
    print('Bases de datos listas para ser utilizadas.')

    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_c, y_c, test_size=0.30, random_state=42)
    X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_d, y_d, test_size=0.30, random_state=42)
    X_train_nc, X_val_nc, y_train_nc, y_val_nc = train_test_split(X_c, y_c, test_size=0.30, random_state=42)

    # Preparing data for LSTM
    # Reshape will convert the output in a correct form.
    X_train_c = X_train_c.reshape((X_train_c.shape[0], 1, X_train_c.shape[1]))
    X_val_c = X_val_c.reshape((X_val_c.shape[0], 1, X_val_c.shape[1]))
    X_train_nc = X_train_nc.reshape((X_train_nc.shape[0], 1, X_train_nc.shape[1]))
    X_val_nc = X_val_nc.reshape((X_val_nc.shape[0], 1, X_val_nc.shape[1]))
    X_test_c = test_confirmed.reshape((test_confirmed.shape[0], 1, test_confirmed.shape[1]))
    X_test_nc = test_nc.reshape((test_nc.shape[0], 1, test_nc.shape[1]))

    X_train_d = X_train_d.reshape((X_train_d.shape[0], 1, X_train_d.shape[1]))
    X_val_d = X_val_d.reshape((X_val_d.shape[0], 1, X_val_d.shape[1]))
    X_test_d = test_deaths.reshape((test_deaths.shape[0], 1, test_deaths.shape[1]))

    # Model preparation
    epochs = 10  # times amount the data will pass through the model
    batch_size = 32
    n_features = 1

    # Confirmed cases model

    # stacked LSTM
    model_c = Sequential()
    model_c.add(LSTM(50, activation='relu', input_shape=(n_features, n_steps), return_sequences=True))
    model_c.add(LSTM(150, activation='relu'))
    model_c.add(Dense(1, activation='relu'))

    # model compilation
    model_c.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
                 EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    # model fit
    model_c.fit(X_train_c, y_train_c, epochs=epochs, batch_size=batch_size, validation_data=(X_val_c, y_val_c),
                verbose=2,
                shuffle=True, callbacks=callbacks)

    # Model for death cases
    model_d = Sequential()
    model_d.add(LSTM(50, activation='relu', input_shape=(n_features, n_steps), return_sequences=True))
    model_d.add(LSTM(50, activation='relu'))
    model_d.add(Dense(1))

    # model compilation
    model_d.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
                 EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    # model fit
    model_d.fit(X_train_d, y_train_d, epochs=epochs, batch_size=batch_size, validation_data=(X_val_d, y_val_d),
                verbose=2,
                shuffle=True, callbacks=callbacks)

    # Prediction model for new confirmed cases.
    model_nc = Sequential()
    model_nc.add(LSTM(50, activation='relu', input_shape=(n_features, n_steps), return_sequences=True))
    model_nc.add(LSTM(50, activation='relu'))
    model_nc.add(Dense(1))
    model_nc.summary()

    # Model compilation
    model_nc.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
                 EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    # Model fit
    model_nc.fit(X_train_nc, y_train_nc, epochs=epochs, batch_size=batch_size,
                 validation_data=(X_val_nc, y_val_nc), verbose=2,
                 shuffle=True, callbacks=callbacks)

    # Prediction
    res_confirmed = forcast(model_c, X_test_c, forcast_start_date, num_days=50, steps=n_steps, regs=regs)
    res_deaths = forcast(model_d, X_test_d, forcast_start_date, num_days=50, steps=n_steps, regs=regs)
    res_new_confirmed = forcast(model_nc, X_test_nc, forcast_start_date, num_days=50, steps=n_steps, regs=regs)

    sub, res_regional = prepare_submission(res_confirmed, res_deaths, res_new_confirmed, test, pivot_date)
    sub.to_csv('submission.csv', index=None)
    print(sub.head())

    result = []

    for country in countries:
        country_name = country.get('name')
        days = country.get('days')

        country_data = {
            'country_name': country_name,
            'confirmed_cases': {},
            'death_cases': {}
        }

        x_truth, y_truth, x_pred, y_pred = get_countrywise_forcast_(covid_timeseries, pivot_date,
                                                                    'New Confirmed', country_name, 'NoState', days,
                                                                    res_regional)

        x_truth, y_truth, x_pred, y_pred = get_countrywise_forcast(train, pivot_date, country_name, 'NoState', days,
                                                                   res_regional)

        covid_timeseries = train.groupby(['ObservationDate', 'Country/Region', 'Province/State'])[
            'Confirmed', 'Deaths'].sum()
        covid_timeseries = covid_timeseries.reset_index().sort_values('ObservationDate')
        covid_timeseries = get_newly_added(covid_timeseries)

        country_data['confirmed_cases'] = {
            'x_truth': x_truth,
            'y_truth': y_truth,
            'x_pred': x_pred,
            'y_pred': y_pred
        }

        x_truth_death, y_truth_death, x_pred_death, y_pred_death = get_countrywise_forcast_deaths(train, pivot_date,
                                                                                                  country_name,
                                                                                                  'NoState',
                                                                                                  days, res_regional)

        country_data['death_cases'] = {
            'x_truth': x_truth_death,
            'y_truth': y_truth_death,
            'x_pred': x_pred_death,
            'y_pred': y_pred_death
        }

        result.append(country_data)

    return result
