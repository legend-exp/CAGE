#!/usr/bin/env python3
import pandas as pd
from io import StringIO

def main():
    """
    """
    # https://elog.legend-exp.org/UWScanner/249 (one file per run)
    tb0 = """V_pulser  run  E_keV  mV_firststage

    """

    # https://elog.legend-exp.org/UWScanner/294
    tb1 = StringIO("""
    V_pulser  run  E_keV  mV_firststage
    3.76  1172  1460  316
    0.05  1173  15    7.2
    0.1   1174  31    11.0
    0.2   1175  62    19.4
    0.5   1176  167   44.0
    0.8   1177  277   69.6
    1     1178  352   85.6
    2     1179  744   172
    5     1180  1971  500
    8     1181  3225  740
    10    1182  4054  880
    """)
    df_puls = pd.read_csv(tb1, delim_whitespace=True)

    print(df_puls)


if __name__=="__main__":
    main()
