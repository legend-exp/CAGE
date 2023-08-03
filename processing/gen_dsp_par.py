import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import os
import pint
import json
import argparse

from pygama.flow import DataLoader
from pygama.flow import FileDB
from lgdo.lh5_store import LH5Store
from lgdo import ls, Table, WaveformTable
from dspeed import build_dsp

def main():
    doc="""
        Generate optimized DSP parameters for UW test stands
    """
    # Command line arguments
    
    rthf = argparse.RawTextHelpFormatter
    par = argparse.ArgumentParser(description=doc, formatter_class=rthf)
    arg, st, sf = par.add_argument, 'store_true', 'store_false'

    arg('--fdb', nargs=1, type=str, help='LH5 file containing FileDB')
    arg('--dl', nargs=1, type=str, help='config file for DataLoader')
    arg('--cyc', nargs=1, type=int, help='the cycle to optimize for')
    arg('-b', '--batch', action=st, help='do not ask for user input on 1460 line and saving')
    arg('--raw', nargs='?', default="1460raw_temp.lh5", type=str, help='temporary raw file for 1460 kev waveforms')
    arg('--dsp', nargs='?', default="1460dsp_temp.lh5", type=str, help='temporary dsp file for optimization')
    arg('--config_dir', nargs='?', default="metadata/dsp", type=str, help='directory to store config file')

    args = par.parse_args()
    
    print("Parsing arguments")
    
    fdb = FileDB(str(args.fdb[0]))
    dl = DataLoader(config=str(args.dl[0]), filedb=fdb)
    cyc = int(args.cyc[0])
    config_dir = str(args.config_dir)
    print(config_dir)
    
    print("Setting up DSP config")
    # Set up DSP config and DB
    dsp_db = {
        "40K": {
            "etrap": {
                "rise": "8*us",
                "flat": "2*us"
            },
            "pz": {
                "tau": "70*us"
            },
            "dcr_trap": {
                "rise": "8*us",
                "flat": "2*us"
            },
            "ctc": {
                "alpha": 1
            }
        }
    }
    dsp_config = {
      "outputs": [
        "trapEmax", "lt_slope"
      ],
      "processors":{
        "bl, bl_sig, bl_slope, bl_intercept":{
          "function": "linear_slope_fit",
          "module": "pygama.dsp.processors",
          "args" : ["waveform[0: 3500]", "bl","bl_sig", "bl_slope","bl_intercept"],
          "unit": ["ADC","ADC","ADC","ADC"]
        },
        "wf_blsub":{
          "function": "subtract",
          "module": "numpy",
          "args": ["waveform", "bl", "wf_blsub"],
          "prereqs": ["waveform", "bl"],
          "unit": "ADC"
        },
        "wf_logtail": {
          "function": "log",
          "module": "numpy",
          "args": ["wf_blsub[4250:8000]", "wf_logtail"],
          "unit": "ADC",
          "prereqs": ["wf_blsub"]
        },
        "lt_mean, lt_sig, lt_slope, lt_int": {
            "function": "linear_slope_fit",
            "module": "pygama.dsp.processors",
            "args": ["wf_logtail", "lt_mean", "lt_sig", "lt_slope", "lt_int"],
            "unit": ["ADC", "ADC", "ADC", "ADC"],
            "prereqs": ["wf_logtail"]
        },
        "wf_pz": {
          "function": "pole_zero",
          "module": "pygama.dsp.processors",
          "args": ["wf_blsub", "db.pz.tau", "wf_pz"],
          "prereqs": ["wf_blsub"],
          "unit": "ADC",
        },
        "wf_etrap": {
          "function": "trap_norm",
          "module": "pygama.dsp.processors",
          "prereqs": ["wf_pz"],
          "args": ["wf_pz", "db.etrap.rise", "db.etrap.flat", "wf_etrap"],
          "unit": "ADC"
        },
        "trapEmax": {
          "function": "amax",
          "module": "numpy",
          "args": ["wf_etrap", 1, "trapEmax"],
          "kwargs": {"signature":"(n),()->()", "types":["fi->f"]},
          "unit": "ADC",
          "prereqs": ["wf_etrap"]
        },
        "trapEmax_ctc": {
          "function": "add",
          "module": "numpy",
          "args": ["trapEmax", "db.ctc.alpha*dcr", "trapEmax_ctc"],
          "unit": "ADC",
          "prereqs": ["trapEmax", "dcr"]
        },
        "tp_min, tp_max, wf_min, wf_max":{
          "function": "min_max",
          "module": "pygama.dsp.processors",
          "args": ["wf_blsub", "tp_min", "tp_max", "wf_min", "wf_max"],
          "unit": ["ns","ns","ADC", "ADC"],
          "prereqs": ["wf_blsub"]
        },
        "pz_mean, pz_sig, pz_slope, pz_int": {
            "function": "linear_slope_fit",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz[5750:8000]", "pz_mean", "pz_sig", "pz_slope", "pz_int"],
            "unit": ["ADC", "ADC", "ADC", "ADC"],
            "prereqs": ["wf_pz"]
        },
        "wf_dcr_trap": {
            "function": "trap_norm",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "db.dcr_trap.rise", "db.dcr_trap.flat", "wf_dcr_trap"],
            "defaults" : {"db.dcr_trap.rise":"7*us", "db.dcr_trap.flat":"20*us"},
            "unit": "ADC",
            "prereqs": ["wf_pz"]
        },
        "dcr": {
            "function": "fixed_time_pickoff",
            "module": "pygama.dsp.processors",
            "args": ["wf_dcr_trap", "db.dcr.ftp", "i", "dcr"],
            "defaults" : {"db.dcr.ftp" : "80*us"},
            "unit": "ADC",
            "prereqs": ["wf_dcr_trap"]
        }
      }
    }
    
    # Load waveforms for 1460 keV peak and write to raw_file
    dl.set_files(f"cycle == {cyc}")
    dl.set_output(fmt="pd.DataFrame", columns=["energy"])
    el = dl.build_entry_list(save_output_columns=True)
    data = dl.load(el)
    
    if not args.batch:
        print("Pick out the 1460 keV peak")
    else:
        print("In batch mode...")
    
    plt.figure()
    k40_hist, k40_bins, _ = plt.hist(data['energy'], bins = np.linspace(1.7e6, 2e6, 100))
    plt.xlabel('energy')
    plt.ylabel('count')
    plt.title('1460 keV peak')
    plt.savefig('./plots/dsp/1460peak.png')
    
    
    if not args.batch:
        elo = float(input("Lower bound of the 1460 peak: "))
        ehi = float(input("Upper bound of the 1460 peak: "))
    else:
        peak_idx = np.argmax(k40_hist)
        elo = k40_bins[peak_idx - 5]
        ehi = k40_bins[peak_idx + 5]
        print(f"Using elo={elo}, ehi={ehi}")
    
    dl.reset()
    dl.set_files(f"cycle == {cyc}")
    dl.set_cuts({"hit": f"energy > {elo} and energy < {ehi}"})
    dl.set_output(columns=["waveform"])
    wfs = dl.load()
    
    raw_file = args.raw
    dsp_file = args.dsp
        
    sto = LH5Store()
    sto.write_object(obj=wfs, name="40K", lh5_file=raw_file, wo_mode="of")
    
    ureg = pint.UnitRegistry()
    per_unit = 1/(wfs['waveform']['dt'].nda[0] * ureg(wfs['waveform']['dt'].attrs['units']).units)
    per_us = per_unit.to("1/us")
    
    print("Getting initial guesses")
    ebins, tau_range, rise_range, dcrrise_range, dcrflat_range, alpha_range = first_guesses(raw_file=raw_file, 
                                                                                              dsp_file=dsp_file, 
                                                                                              dsp_config=dsp_config, 
                                                                                              dsp_db=dsp_db, 
                                                                                              per_us=per_us.magnitude)
    dsp_config["processors"].pop("wf_logtail")
    dsp_config["processors"].pop("lt_mean, lt_sig, lt_slope, lt_int")
    dsp_config["outputs"] = ["pz_slope", "trapEmax", "dcr", "trapEmax_ctc"]

    print("Optimizing pole zero...")
    dsp_db = opt_pole_zero(tau_range, raw_file, dsp_file, dsp_config, dsp_db)
    print("Optimizing energy trapezoid...")
    dsp_db = opt_etrap(rise_range, raw_file, dsp_file, dsp_config, dsp_db, ebins)
    print("Optimizing DCR...")
    dsp_db = opt_dcrtrap(dcrrise_range, dcrflat_range, raw_file, dsp_file, dsp_config, dsp_db)
    print("Optimizing charge trapping correction...")
    dsp_db = opt_ctc(alpha_range, raw_file, dsp_file, dsp_config, dsp_db, ebins)
    
    print("Checking final results")
    check_final(cyc, fdb, dsp_config, dsp_db, config_dir, args.batch)    
    
    
    
    
def first_guesses(raw_file: str, 
                  dsp_file: str, 
                  dsp_config: dict, 
                  dsp_db: dict, 
                  per_us: int):
    """
    Generate parameter ranges for each of the DSP parameters we want to optimize
    """
    build_dsp(f_raw=raw_file, f_dsp=dsp_file, dsp_config=dsp_config, database=dsp_db, write_mode='r', n_max=100)
    sto = LH5Store()
    pk_table, _ = sto.read_object("40K", dsp_file)
    pk_df = pk_table.get_dataframe()
    
    emed = pk_df.median()['trapEmax']
    ebins = np.linspace(emed - 0.02*emed, emed + 0.02*emed, 20)
    
    lt_tau = 1 / (pk_df['lt_slope'].mean()*per_us) # ADC/sample * samples/us = ADC/us
    tau_range = -np.arange(lt_tau - 1, lt_tau + 1, .2)
    
    rise_range = np.arange(1, 15)
    dcrrise_range = np.arange(2, 5)
    dcrflat_range = np.arange(18, 21)
    alpha_range = np.append([0], np.linspace(0.5, 15, 10))
    
    return ebins, tau_range, rise_range, dcrrise_range, dcrflat_range, alpha_range
    
    
def opt_pole_zero(tau_range, raw_file, dsp_file, dsp_config, dsp_db): 
    optimized = False
    best_tau = 0
    sto = LH5Store()
    while not optimized:
        results = None

        for tau in tau_range:
            dsp_db["40K"]["pz"]["tau"] = str(tau) + " * us"

            build_dsp(f_raw=raw_file, f_dsp=dsp_file, dsp_config=dsp_config, database=dsp_db, write_mode='r', n_max=20)

            pk_table, _ = sto.read_object("40K", dsp_file)
            res = pd.DataFrame({
                "tau": [tau],
                "pz_slope_avg": [np.nanmean(np.abs(pk_table['pz_slope'].nda))]
            })
            if results is None:
                results = res
            else:
                results = pd.concat([results, res], ignore_index=True)
        best_tau = results.abs().sort_values("pz_slope_avg").iloc[0]['tau']
        if best_tau == tau_range[0]:
            tau_range = np.linspace(tau_range[0]-1, tau_range[0]+0.5, 10)
        elif best_tau == tau_range[-1]:
            tau_range = np.linspace(tau_range[-1]-0.5, tau_range[-1]+1, 10)
        else:
            optimized = True
    dsp_db["40K"]["pz"]["tau"] = str(best_tau) + "* us"
    return dsp_db
    
    
def opt_etrap(rise_range, raw_file, dsp_file, dsp_config, dsp_db, ebins): 
    results = None
    best_rise = 0
    sto = LH5Store()
    for rise in rise_range:
        dsp_db["40K"]["etrap"]["rise"] = str(rise) + " * us"

        build_dsp(f_raw=raw_file, f_dsp=dsp_file, dsp_config=dsp_config, database=dsp_db, write_mode='r')

        pk_table, _ = sto.read_object("40K", dsp_file)
        ehist, _ = np.histogram(pk_table['trapEmax'].nda, bins = ebins)

        res = pd.DataFrame({
            "rise": [rise],
            "peak_height": [np.max(ehist)]
        })
        if results is None:
            results = res
        else:
            results = pd.concat([results, res], ignore_index=True)
    best_rise = results.abs().sort_values("peak_height").iloc[-1]['rise']   
    dsp_db["40K"]["etrap"]["rise"] = str(best_rise) + "* us"
    return dsp_db
    
def opt_dcrtrap(dcrrise_range, dcrflat_range, raw_file, dsp_file, dsp_config, dsp_db):
    results = None
    optimized = False
    best_dcrrise = 0
    best_dcrflat = 0
    sto = LH5Store()
    while not optimized:
        for rise in dcrrise_range:
            for flat in dcrflat_range:
                dsp_db["40K"]["dcr_trap"]["rise"] = str(rise) + " * us"
                dsp_db["40K"]["dcr_trap"]["flat"] = str(flat) + " * us"

                build_dsp(f_raw=raw_file, f_dsp=dsp_file, dsp_config=dsp_config, database=dsp_db, write_mode='r')

                pk_table, _ = sto.read_object("40K", dsp_file)

                res = pd.DataFrame({
                    "rise": [rise],
                    "flat": [flat],
                    "dcr_mean": [np.abs(np.mean(pk_table['dcr'].nda))]
                })
                if results is None:
                    results = res
                else:
                    results = pd.concat([results, res], ignore_index=True)
        best_dcrrise = results.abs().sort_values("dcr_mean").iloc[0]['rise']
        best_dcrflat = results.abs().sort_values("dcr_mean").iloc[0]['flat']
        
        if (best_dcrrise == dcrrise_range[0] or
                best_dcrrise == dcrrise_range[-1] or
                best_dcrflat == dcrflat_range[0] or
                best_dcrflat == dcrflat_range[-1]):
            if best_dcrrise == dcrrise_range[0]:
                dcrrise_range = np.linspace(dcrrise_range[0]-0.75, dcrrise_range[0]+0.25, 4)
            if best_dcrrise == dcrrise_range[-1]:
                dcrrise_range = np.linspace(dcrrise_range[-1]-0.25, dcrrise_range[-1]+0.75, 4)
            if best_dcrflat == dcrflat_range[0]:
                dcrflat_range = np.linspace(dcrflat_range[0]-0.75, dcrflat_range[0]+0.25, 4)
            if best_dcrflat == dcrflat_range[-1]:
                dcrflat_range = np.linspace(dcrflat_range[-1]-0.25, dcrflat_range[-1]+0.75, 4)
        else:
            optimized = True
    

    dsp_db["40K"]["dcr_trap"]["rise"] = str(best_dcrrise) + "* us"
    dsp_db["40K"]["dcr_trap"]["flat"] = str(best_dcrflat) + "* us"
    return dsp_db
                
def opt_ctc(alpha_range, raw_file, dsp_file, dsp_config, dsp_db, ebins): 
    optimized = False
    best_alpha = 0
    sto = LH5Store()
    while not optimized:
        results = None
        for alpha in alpha_range:
            dsp_db["40K"]["ctc"]["alpha"] = str(alpha)
            build_dsp(f_raw=raw_file, f_dsp=dsp_file, dsp_config=dsp_config, database=dsp_db, write_mode='r')

            pk_table, _ = sto.read_object("40K", dsp_file)
            ehist, _ = np.histogram(pk_table['trapEmax_ctc'].nda, bins = ebins)

            res = pd.DataFrame({
                "alpha": [alpha],
                "peak_height": [np.max(ehist)]
            })
            if results is None:
                results = res
            else:
                results = pd.concat([results, res], ignore_index=True)
        best_alpha = results.abs().sort_values("peak_height").iloc[-1]['alpha']
        if best_alpha == 0 and alpha_range[-1] > 1:
            alpha_range = np.linspace(0, 1, 10)
        elif best_alpha == alpha_range[-1]:
            alpha_range = np.linspace(alpha_range[-1]-0.5, alpha_range[-1]+5, 10)
        elif alpha_range[-1]-alpha_range[-2] > 0.5:
            print(alpha_range)
            alpha_range = np.linspace(best_alpha-0.5, best_alpha+0.5, 10)
        else:
            optimized = True
    
    dsp_db["40K"]["ctc"]["alpha"] = best_alpha
    return dsp_db
    
    
def check_final(cyc, fdb, dsp_config, dsp_db, config_dir, batch):
    raw = fdb.df.query(f"cycle == {cyc}").iloc[0]
    raw = os.path.join(fdb.data_dir, fdb.tier_dirs['raw'], raw['raw_file'])
    dsp = f"cycle{cyc}_testdsp.lh5"
    
    sto = LH5Store()
    
    with open('./metadata/dsp/dsp_07.json') as f:
        test_config = json.load(f)
    test_config['processors']['wf_pz']['defaults'] = {"db.pz.tau": dsp_db['40K']['pz']['tau']}
    test_config['processors']['wf_etrap']['defaults'] = {"db.etrap.rise": dsp_db['40K']['etrap']['rise'], 
                                                      "db.etrap.flat": dsp_db['40K']['etrap']['flat']}
    test_config['processors']['wf_dcr_trap']['defaults'] = {"db.dcr_trap.rise": dsp_db['40K']['dcr_trap']['rise'], 
                                                      "db.dcr_trap.flat": dsp_db['40K']['dcr_trap']['flat']}
    test_config['processors']['trapEmax_ctc']['defaults'] = {"db.ctc.alpha": dsp_db['40K']['ctc']['alpha']}
    
    print("Found values: ")
    print(dsp_db)
    
    if batch:
        print(f"Writing config to: {config_dir}/dsp_cyc{cyc}.json")
        with open(f'{config_dir}/dsp_cyc{cyc}.json', 'w') as f:
            json.dump(test_config, f)
    else:
        build_dsp(f_raw=raw, f_dsp=dsp, dsp_config=test_config, write_mode='r')

        dsp_table, _ = sto.read_object("ORSIS3302DecoderForEnergy/dsp", dsp)

        ehist, trapEbins = np.histogram(dsp_table['trapEmax'].nda, bins=np.arange(2500, 4000))
        k40_peak = trapEbins[np.argmax(ehist)]

        plt.figure()
        plt.yscale('log')
        plt.hist(dsp_table['trapEmax'].nda*(1460/k40_peak), bins=np.arange(1450, 2620))
        plt.xlabel('trapEmax, scaled to 1460 peak')
        plt.ylabel('count')
        plt.title('Check the linearity of 2615 keV peak')
        plt.savefig('./plots/dsp/linearity.png')

        plt.figure()
        plt.hist(dsp_table['dcr'].nda, bins=np.arange(-50, 50))
        plt.xlabel('dcr')
        plt.ylabel('count')
        plt.title('Check the DCR')
        plt.savefig('./plots/dsp/dcr.png')

        plt.figure()
        plt.hist2d(dsp_table['trapEmax'].nda, dsp_table['dcr'].nda, 
                   bins = (np.linspace(0, 10000, 100), np.arange(-200, 200)), 
                   norm=colors.LogNorm())
        plt.xlabel("trapEmax")
        plt.ylabel("DCR")
        plt.title("trapEmax vs. DCR")
        plt.axhline(0, color='r')
        plt.savefig('./plots/dsp/trapEmax_dcr.png')

        plt.figure()
        plt.hist2d(dsp_table['trapEmax_ctc'].nda, dsp_table['dcr'].nda, 
                   bins = (np.linspace(0, 10000, 100), np.arange(-200, 200)), 
                   norm=colors.LogNorm())
        plt.xlabel("trapEmax_ctc")
        plt.ylabel("DCR")
        plt.title("trapEmax_ctc vs. DCR")
        plt.axhline(0, color='r')
        plt.savefig('./plots/dsp/trapEmaxctc_dcr.png')
    
        save = input("Should we save this configuration? y/n: ")
        if save.strip().lower() == "y":
            print(f"Writing config to: {config_dir}/dsp_cyc{cyc}.json")
            with open(f'{config_dir}/dsp_cyc{cyc}.json', 'w') as f:
                json.dump(test_config, f)
            
if __name__ == "__main__":
    main()
