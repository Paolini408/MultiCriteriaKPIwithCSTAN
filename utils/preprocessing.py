import numpy as np
import pandas as pd
import yaml

with open('./configs/config.yml', 'r') as file:
    config = yaml.safe_load(file)


def quantify_missing_values(dataframe):
    return dataframe.isnull().sum()


def fill_missing_values_with_mean(dataframe):
    for column in dataframe.columns:
        if dataframe[column].isnull().any():
            mean_value = dataframe[column].mean()
            dataframe[column] = dataframe[column].fillna(mean_value)
    return dataframe


def fahrenheit_to_celsius(T_fahrenheit):
    T_celsius = (T_fahrenheit - 32) * 5 / 9
    return T_celsius


def GPM_to_kgs(water_flow_GPM):
    water_flow_kgs = water_flow_GPM * 63 / 1000
    return water_flow_kgs


def CFM_to_m3h(air_flow_CFM):
    air_flow_m3h = air_flow_CFM * 1.6990107955
    return air_flow_m3h


def revs_to_rpm(revs_speed):
    rpm_speed = revs_speed * 60
    return rpm_speed


def change_unit(dataframe):
    temperature_to_convert = config['columns_to_convert']['temp']
    water_flow_to_convert = config['columns_to_convert']['water_flow']
    air_flow_to_convert = config['columns_to_convert']['air_flow']
    speed_to_convert = config['columns_to_convert']['fan_speed']

    for column in temperature_to_convert:
        if column in dataframe.columns:
            if dataframe[column].dtype != 'object':
                dataframe[column] = fahrenheit_to_celsius(dataframe[column])
            else:
                print(f'Column {column} is not numeric')

    for column in water_flow_to_convert:
        if column in dataframe.columns:
            if dataframe[column].dtype != 'object':
                dataframe[column] = GPM_to_kgs(dataframe[column])
            else:
                print(f'Column {column} is not numeric')

    for column in air_flow_to_convert:
        if column in dataframe.columns:
            if dataframe[column].dtype != 'object':
                dataframe[column] = CFM_to_m3h(dataframe[column])
            else:
                print(f'Column {column} is not numeric')

    for column in speed_to_convert:
        if column in dataframe.columns:
            if dataframe[column].dtype != 'object':
                dataframe[column] = revs_to_rpm(dataframe[column])
            else:
                print(f'Column {column} is not numeric')
    return dataframe


def obtain_power_coils(dataframe):
    cp = 4186  # J/kgK
    m_cooling = dataframe[config['columns_to_convert']['water_flow'][0]]
    m_heating = dataframe[config['columns_to_convert']['water_flow'][1]]
    dt_cooling = abs(
        dataframe[config['columns_to_convert']['temp'][7]] - dataframe[config['columns_to_convert']['temp'][8]])
    dt_heating = abs(
        dataframe[config['columns_to_convert']['temp'][9]] - dataframe[config['columns_to_convert']['temp'][10]])
    dataframe['P_cooling'] = m_cooling * cp * dt_cooling  # W
    dataframe['P_heating'] = m_heating * cp * dt_heating  # W
    return dataframe


def get_time_info(dataframe):
    dataframe['Datetime'] = pd.to_datetime(dataframe['Datetime'], errors='coerce')
    dataframe['Year'] = dataframe['Datetime'].dt.year
    dataframe['Month'] = dataframe['Datetime'].dt.month
    dataframe['Day'] = dataframe['Datetime'].dt.day
    dataframe['Hour'] = dataframe['Datetime'].dt.hour
    dataframe['DayOfWeek'] = dataframe['Datetime'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    return dataframe


def resample_15_min(dataframe, time_interval='15min'):
    df_resampled = dataframe.resample(time_interval, on='Datetime').mean(numeric_only=True)
    df_resampled.reset_index(inplace=True)
    return df_resampled


def obtain_control_sequence(dataframe):
    dataframe['Cooling_state'] = 0
    dataframe['Heating_state'] = 0

    T_room = config['t_room']
    T_cooling_setpoint = config['t_cooling_sp']
    T_heating_setpoint = config['t_heating_sp']

    cooling_coil = 0
    heating_coil = 0

    for index, row in dataframe.iterrows():
        if cooling_coil == 0:
            if row[T_room] <= (row[T_cooling_setpoint] + 5 / 9):
                dataframe.at[index, 'Cooling_state'] = 0
            else:
                cooling_coil = True
                dataframe.at[index, 'Cooling_state'] = 1

        if cooling_coil == 1:
            if row[T_room] >= (row[T_cooling_setpoint] - 5 / 9):
                dataframe.at[index, 'Cooling_state'] = 1
            else:
                cooling_coil = False
                dataframe.at[index, 'Cooling_state'] = 0

    for index, row in dataframe.iterrows():
        if heating_coil == 0:
            if row[T_room] >= (row[T_heating_setpoint] - 5 / 9):
                dataframe.at[index, 'Heating_state'] = 0
            else:
                heating_coil = True
                dataframe.at[index, 'Heating_state'] = 1

        if heating_coil == 1:
            if row[T_room] <= (row[T_heating_setpoint] + 5 / 9):
                dataframe.at[index, 'Heating_state'] = 1
            else:
                heating_coil = False
                dataframe.at[index, 'Heating_state'] = 0
    return dataframe


def set_on_off(dataframe):
    if 'Cooling_state' in dataframe.columns and 'Heating_state' in dataframe.columns:
        dataframe['Cooling_state'] = dataframe['Cooling_state'].apply(lambda x: 'On' if x >= 0.5 else 'Off')
        dataframe['Heating_state'] = dataframe['Heating_state'].apply(lambda x: 'On' if x >= 0.5 else 'Off')
    else:
        raise KeyError("ERROR: 'Cooling_state' and/or 'Heating_state' columns not found in dataframe")
    return dataframe


def obtain_operational_mode(dataframe):
    dataframe['Operational_mode'] = None
    for index, row in dataframe.iterrows():
        if row[config['fcu_mode']] < 1:
            dataframe.at[index, 'Operational_mode'] = 'Shutdown mode'
        if 1 <= row[config['fcu_mode']] < 2:
            if row['Cooling_state'] == 'On' and row['Heating_state'] == 'Off':
                dataframe.at[index, 'Operational_mode'] = 'Cooling mode'
            if row['Cooling_state'] == 'Off' and row['Heating_state'] == 'On':
                dataframe.at[index, 'Operational_mode'] = 'Heating mode'
            if row['Cooling_state'] == 'Off' and row['Heating_state'] == 'Off':
                dataframe.at[index, 'Operational_mode'] = 'Off mode'
            if row['Cooling_state'] == 'On' and row['Heating_state'] == 'On':
                dataframe.at[index, 'Operational_mode'] = 'Simultaneous cc/hc'
        if row[config['fcu_mode']] == 2:
            if row['Cooling_state'] == 'On' and row['Heating_state'] == 'Off':
                dataframe.at[index, 'Operational_mode'] = 'Setback cooling mode'
            if row['Cooling_state'] == 'Off' and row['Heating_state'] == 'On':
                dataframe.at[index, 'Operational_mode'] = 'Setback heating mode'
            if row['Cooling_state'] == 'Off' and row['Heating_state'] == 'Off':
                dataframe.at[index, 'Operational_mode'] = 'Setback off mode'
            if row['Cooling_state'] == 'On' and row['Heating_state'] == 'On':
                dataframe.at[index, 'Operational_mode'] = 'Setback simultaneous cc/hc'
    return dataframe


def get_steady(dataframe):
    transient_cutoff = config['transient_cutoff']
    df_steady = dataframe.copy()

    slope_cooling = pd.Series(
        np.gradient(df_steady[config['ccv_command']]),
        df_steady.index,
        name='slope_cooling')

    slope_heating = pd.Series(
        np.gradient(df_steady[config['hcv_command']]),
        df_steady.index,
        name='slope_heating')

    slope_damper_out = pd.Series(
        np.gradient(df_steady[config['oad_command']]),
        df_steady.index,
        name='slope_damper_out')

    slope = pd.concat([slope_cooling, slope_heating, slope_damper_out], axis=1)
    view_slopes = slope.copy()
    slope = slope.abs().max(axis=1)
    df_steady = pd.concat([df_steady, slope], axis=1)
    df_steady = df_steady.rename(columns={0: 'slope'})

    df_steady['slope'] = np.where(
        df_steady['slope'] > transient_cutoff,
        'transient',
        'steady'
    )
    return df_steady, view_slopes


def remove_transition(dataframe):
    idx_list = []
    for i in range(1, len(dataframe)):
        if dataframe.iloc[i]['Operational_mode'] != dataframe.iloc[i - 1]['Operational_mode']:
            idx_list.append(i)
    dataframe = dataframe.drop(dataframe.index[idx_list])
    return dataframe


def get_labels_detection(dataframe, name_df):
    dataframe['label_detection'] = None
    if name_df == 'FCU_FaultFree':
        dataframe['label_detection'] = 'Normal old'
    # BIAS #
    elif name_df.startswith('FCU_SensorBias_RMTemp'):
        dataframe['label_detection'] = 'Fault'
    # OUTLET BLOCKAGE and FILTERS #
    elif name_df.startswith('FCU_FilterRestriction') or name_df.startswith('FCU_FanOutletBlockage'):
        dataframe['label_detection'] = 'Fault'
    # DAMPER AREA (LEAK) and BLOCKAGE #
    elif name_df.startswith('FCU_OADMPRLeak') or name_df.startswith('FCU_OABlockage'):
        dataframe['label_detection'] = 'Fault'
    # CONTROL SYSTEM #
    elif name_df.startswith('FCU_Control'):
        dataframe['label_detection'] = 'Fault'
    # FOULING AIR (both heating and cooling) #
    elif name_df.startswith('FCU_Fouling_Cooling_Airside') or name_df.startswith('FCU_Fouling_Heating_Airside'):
        dataframe['label_detection'] = 'Fault'
    # FOULING WATER #
    elif name_df.startswith('FCU_Fouling_Cooling_Waterside'):
        for index, row in dataframe.iterrows():
            if row['P_cooling'] < 50:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_Fouling_Heating_Waterside'):
        for index, row in dataframe.iterrows():
            if row['P_heating'] < 50:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    # STUCK #
    elif name_df.startswith('FCU_VLVStuck_Cooling'):
        for index, row in dataframe.iterrows():
            if row[config['ccv_position']] - 0.02 < row[config['ccv_command']] < row[config['ccv_position']] + 0.02:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_VLVStuck_Heating'):
        for index, row in dataframe.iterrows():
            if row[config['hcv_position']] - 0.02 < row[config['hcv_command']] < row[config['hcv_position']] + 0.02:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_OADMPRStuck'):
        for index, row in dataframe.iterrows():
            if row[config['oad_position']] - 0.02 < row[config['oad_command']] < row[config['oad_position']] + 0.02:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    # LEAK #
    elif name_df.startswith('FCU_VLVLeak_Cooling_20'):
        for index, row in dataframe.iterrows():
            if row[config['cooling_water_flow']] > (0.2*0.36)*1.05:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_VLVLeak_Cooling_50'):
        for index, row in dataframe.iterrows():
            if row[config['cooling_water_flow']] > (0.5*0.36)*1.05:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_VLVLeak_Cooling_80'):
        for index, row in dataframe.iterrows():
            if row[config['cooling_water_flow']] > (0.8*0.36)*1.05:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_VLVLeak_Heating_20'):
        for index, row in dataframe.iterrows():
            if row[config['heating_water_flow']] > (0.2*0.066)*1.05:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_VLVLeak_Heating_50'):
        for index, row in dataframe.iterrows():
            if row[config['heating_water_flow']] > (0.5*0.066)*1.05:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    elif name_df.startswith('FCU_VLVLeak_Heating_80'):
        for index, row in dataframe.iterrows():
            if row[config['heating_water_flow']] > (0.8*0.066)*1.05:
                dataframe.at[index, 'label_detection'] = 'Normal new'
            else:
                dataframe.at[index, 'label_detection'] = 'Fault'
    return dataframe


def get_labels_isolation(dataframe, name_df):
    dataframe['label_isolation'] = None

    if name_df == 'FCU_FaultFree':
        base_label = 'Normal'
    elif name_df.startswith('FCU_OADMPR') or name_df.startswith('FCU_OABlockage'):
        base_label = 'Fault OAD'
    elif name_df.startswith('FCU_VLVLeak_Cooling') or name_df.startswith('FCU_VLVStuck_Cooling'):
        base_label = 'Fault CC'
    elif name_df.startswith('FCU_VLVLeak_Heating') or name_df.startswith('FCU_VLVStuck_Heating'):
        base_label = 'Fault HC'
    elif name_df.startswith('FCU_SensorBias_RMTemp'):
        base_label = 'Fault RM_TEMP bias'
    elif name_df.startswith('FCU_FilterRestriction') or name_df.startswith(
            'FCU_FanOutletBlockage') or name_df.startswith('FCU_Fouling_Cooling_Airside') or name_df.startswith(
            'FCU_Fouling_Heating_Airside'):
        base_label = 'Fault Air Duct system'
    elif name_df.startswith('FCU_Fouling_Cooling_Waterside') or name_df.startswith('FCU_Fouling_Heating_Waterside'):
        base_label = 'Fault Piping system'
    elif name_df.startswith('FCU_Control'):
        base_label = 'Fault Control system'
    else:
        base_label = None

    def determine_label(row):
        if row['label_detection'] in ['Normal old', 'Normal new']:
            return row['label_detection']
        else:
            return base_label

    dataframe['label_isolation'] = dataframe.apply(determine_label, axis=1)
    return dataframe


def get_labels_diagnosis(dataframe, name_df):
    def determine_label(row):
        if row['label_detection'] in ['Normal old', 'Normal new']:
            return row['label_detection']
        elif row['label_detection'] == 'Fault':
            return name_df

    dataframe['label_diagnosis'] = dataframe.apply(determine_label, axis=1)
    return dataframe
