#!/usr/bin/env python3
import json
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use("clint.mpl")
import matplotlib.dates as md

# silence annoying warning about plotting pd.datetime objects with mpl
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def main():
    """
    An example of using the psycopg2 module to pull values from the CAGE DB.
    
    GOAL: 
    compare the thermal cycle on 9/30 which showed an outgassing event,
    to the previous thermal cycle on 8/30 which was done w/ the top hat & 
    motor proxy.
        -- 9/30 elog: https://elog.legend-exp.org/UWScanner/68
        -- 8/30 elog: https://elog.legend-exp.org/UWScanner/59
    
    TODO:
    Calculate warmup rate for diff. configurations
    """
    
    # objects to access the CAGE DB.  make them global because i'm lazy
    global db_conn, db_cursor

    with open("config.json") as f:
        config = json.load(f)

    db_conn = psycopg2.connect(dbname=config["db_name"], 
                               user=config["db_user"], 
                               password=config["password"], 
                               host=config["cage_daq"])
    db_cursor = db_conn.cursor()
    
    # -- run analysis -- 
    # get_endpoints()
    # get_cooldown_data()
    # plot_cooldown_data()
    
    get_temp()
    

def get_endpoints():
    """
    grab the current list of endpoints
    """
    cmd = "SELECT * FROM endpoint_id_map;"
    db_cursor.execute(cmd)
    record = db_cursor.fetchall()
    df_end = pd.DataFrame(record, columns=["row","endpoint_name","data_type"])
    print(df_end)
    
    
def pandas_db_query(endpoints, t_earlier, t_later, df_file=None):
    """
    general use function.
    for each endpoint and start/stop dates, create a DataFrame.
    if `df_file` is set, write the output to an HDF5 file.
    """
    dfs = {}

    for pt in endpoints:
        query = f"SELECT value_cal, timestamp FROM numeric_data "
        query += f"WHERE endpoint_name='{pt}'"
        query += f"AND timestamp>='{t_earlier}' and timestamp<='{t_later}';"
        
        db_cursor.execute(query)
        record = db_cursor.fetchall()
        df = pd.DataFrame(record, columns=[pt, 'timestamp']) 
        
        # print(pt)
        # print(dfs[pt].head(10))
        
        dfs[pt] = df
        
        if df_file is not None:
            df.to_hdf(df_file, key=pt)
            
    return dfs
    
    
def get_cooldown_data():
    """
    8/30 note from Joule:
    It took about 12 hours to reach the lowest temperature. The first 6 hours 
    of cooling used about 5 kg of LN, and it seems like the average LN useage 
    was 4 kg/day after that (Attachment 4).

    can try the automatic x-axis date formatting:
    https://matplotlib.org/3.1.0/gallery/ticks_and_spines/date_concise_formatter.html
    https://matplotlib.org/3.1.0/api/dates_api.html#matplotlib.dates.AutoDateFormatter
    """
    endpoints = ['cage_pressure', 'cage_coldPlate_temp', 'cage_ln_level', 
                 'cage_motor_temp', 'cage_topHat_temp']
    
    df_file = "./plots/cooldown_aug.h5"
    t_earlier_aug = '2019-08-31T00:00'
    t_later_aug = '2019-09-05T00:00'
    pandas_db_query(endpoints, t_earlier_aug, t_later_aug, df_file)
    
    df_file = "./plots/cooldown_sep.h5"
    t_earlier_sep = '2019-09-26T00:00'
    t_later_sep = '2019-10-01T00:00'
    pandas_db_query(endpoints, t_earlier_sep, t_later_sep, df_file)


def plot_cooldown_data():
    """
    """
    df_file = "./plots/cooldown_aug.h5"
    # df_file = "./plots/cooldown_sep.h5"
    
    with pd.HDFStore(df_file, 'r') as f:
        print("Keys:", f.keys())
    
    ln_level = pd.read_hdf(df_file, key='/cage_ln_level')
    motor_temp = pd.read_hdf(df_file, key='/cage_motor_temp')
    tophat_temp = pd.read_hdf(df_file, key='/cage_topHat_temp')
    cp_temp = pd.read_hdf(df_file, key='/cage_coldPlate_temp')
    pressure = pd.read_hdf(df_file, key='/cage_pressure')
    
    # plt.plot(ln_level["timestamp"], ln_level["cage_ln_level"], '-b')
    # plt.ylabel("LN Level", color='b', ha='right', y=1)
    
    plt.semilogy(pressure["timestamp"], 
                 pressure["cage_pressure"], '-b')
    plt.ylabel("pressure (hPa)", color='b', ha='right', y=1)
    # plt.ylim(1e-6, 1e-5)
    
    plt.tick_params('y', colors='b')
    plt.gcf().autofmt_xdate() # rotates labels
    
    p1a = plt.gca().twinx()
    p1a.plot(cp_temp["timestamp"], cp_temp["cage_coldPlate_temp"] + 273, "-r")
    p1a.set_ylabel('Cold Plate Temp (K)', color='r', ha='right', y=1)
    p1a.tick_params('y', colors='r')
    
    plt.tight_layout()
    plt.show()
    

def get_temp():
    """
    example of quickly accessing the DB to get something 'now'.
    """
    epts = ["cage_coldPlate_temp", "cage_pressure"]
    # t_earlier_aug = '2019-10-02T00:00'
    # t_later_aug = datetime.utcnow().isoformat()
    t_earlier_aug = '2019-09-27T13:00'
    t_later_aug = '2019-09-28T19:49'
    dfs = pandas_db_query(epts, t_earlier_aug, t_later_aug)
    print(dfs[epts[0]].tail())
    
    exit()

    xv = dfs[epts[0]]["timestamp"]
    yv = dfs[epts[0]][epts[0]]
    plt.plot(xv, yv, '-b')
    plt.ylabel(epts[0], ha='right', y=1)
    
    p1a = plt.gca().twinx()
    xv = dfs[epts[1]]["timestamp"]
    yv = dfs[epts[1]][epts[1]]
    p1a.set_ylabel(epts[1], color='r', ha='right', y=1)
    p1a.tick_params('y', colors='r')
    p1a.semilogy(xv, yv, '-r')
    
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
    
    

if __name__=="__main__":
    main()