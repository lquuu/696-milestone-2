import pandas as pd
from ambiance import Atmosphere
import numpy as np

# Cleaning and Feature Engineering
#  * columns are renamed  so the meaning of the sensor data can be understood.
#  * Inlet mass airflow is defined
#  * Efficiency metrics are calculated

# columns are renamed so the meaning of the sensor data can be understood.

def english_features(df) -> pd.DataFrame:
    name_map = {'hs': 'health_status',
                'Fc': 'flight_cond',
                'TRA': 'throttle_resolver_angle',
                'T2': 'inlet_temp_tot',
                'T24': 'LPC_outlet_temp_tot',
                'T30': 'HPC_outlet_temp_tot',
                'T40': 'burner_outlet_temp_tot',
                'T48': 'HPT_outlet_temp_tot',  # Efficiency indicator
                'T50': 'LPT_outlet_temp_tot',
                'P2': 'inlet_psia_tot',
                'P15': 'bypass_psia_tot',
                'P21': 'fan_outlet_psia_tot',
                'P24': 'LPC_outlet_psia_tot',
                'P30': 'HPC_outlet_psia_tot',
                'Ps30': 'HPC_outlet_psia_static',
                'P40': 'burner_outlet_psia_tot',
                'P45': 'HPT_outlet_psia_tot',
                'P50': 'LPT_outlet_psia_tot',
                'Nf': 'fan_speed',
                'Nc': 'core_speed',
                'Wf': 'fuel_flow',
                'W21': 'fan_flow_pps',
                'W22': 'LPC_outflow_flow',
                'W25': 'HPC_inflow_flow',
                'W31': 'HPT_coolant_bleed_stageA_flow',
                'W32': 'HPT_coolant_bleed_stageB_flow',
                'W50': 'LPT_outflow',
                'Smfan': 'fan_stall_margin',
                'Smlpc': 'LPC_stall_margin',
                'Smhpc': 'HPC_stall_margin',
                'phi': 'fuel_flow_to_HPC_outlet_psia_static_ratio',  # Trim parameter?
                'W48': 'HPT_outlet_flow',  # Flow indicator
                }

    #for col in df.columns:
    #    if col in name_map.keys():
    #        df.rename(columns={col: name_map[col]})

    return df.rename(columns=name_map)

# Inlet mass airflow is defined

def maf(mach, alt_ft):
    """
    Glenn Research Center: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/mass-flow-rate-equations/
    mass_airflow = m_dot = rVA
    r = density (dalton's law at altitude)
    V = velocity (mach number at altitude) V = Ma = M*sqrt(gammaRT)
        M = mach number
        sqrt(gammaRT) = specific heat ratio, gas constant, and temperature dynamic = handled in ambiance library
    A = area
    """
    # 1 ft = 0.3048 m, 1 ft² = 0.09290304 m², and 1 kg = 2.20462262 lb(mass).
    A = 0.09290304  # ft^2 in meters
    atm = Atmosphere(np.asarray(alt_ft * 0.3048))  # altitude in meters
    V = np.asarray(mach) * atm.speed_of_sound  # m/s
    mdot_kg_s = (atm.density * V * A)
    return mdot_kg_s * 2.20462262  # lb/s

# Efficiency metrics are calculated

def include_eff_params(df) -> pd.DataFrame:
    '''
    Caclulate efficiency metrics for each engine stage
    Input: df
    Output: df with new features
    '''
    # Fan flow efficiency
    df['inlet_maf'] = maf(df['Mach'], df['alt'])  # maf is a different module
    df['fan_flow_eff'] = df['fan_flow_pps'] / df['inlet_maf']
    # Fan pressure efficiency
    df['fan_psia_eff'] = df['bypass_psia_tot'] / df['inlet_psia_tot']

    # LPC flow efficiency
    df['lpc_flow_eff'] = df['LPC_outflow_flow'] / df['fan_flow_pps']
    # LPC pressure efficiency
    df['lpc_psia_eff'] = df['LPC_outlet_psia_tot'] / df['fan_outlet_psia_tot']

    # HPC pressure efficiency
    df['hpc_psia_eff'] = df['HPC_outlet_psia_tot'] / df['LPC_outlet_psia_tot']
    # HPC temperature efficiency
    df['hpc_temp_eff'] = df['HPC_outlet_temp_tot'] / df['LPC_outlet_temp_tot']

    # HPT pressure efficiency
    df['hpt_psia_eff'] = df['HPC_outlet_psia_tot'] / df['HPT_outlet_psia_tot']
    # HPT temperature efficiency
    df['hpt_temp_eff'] = df['HPC_outlet_temp_tot'] / df['HPT_outlet_temp_tot']

    # LPT pressure efficiency
    df['lpt_psia_eff'] = df['HPT_outlet_psia_tot'] / df['LPT_outlet_psia_tot']
    # LPT temperature efficiency
    df['lpt_temp_eff'] = df['HPT_outlet_temp_tot'] / df['LPT_outlet_temp_tot']

    return df

def export_dataframe(df):
    english_column_names = english_features(df)
    engineered_eff_included = include_eff_params(english_column_names)
    return engineered_eff_included