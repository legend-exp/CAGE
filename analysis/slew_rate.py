#!/usr/bin/env python3
"""
slew rate measurement:
https://elog.legend-exp.org/UWScanner/303

just plot the rise time vs. input voltage of a pulser (amp side, not HV side)
"""
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
plt.style.use('../clint.mpl')

tb = StringIO("""
V_pulser  RT_stage2_ns  mVpp_stage2
0.500  53  28.2
0.623  54  34.0
1.000  54  55.2
2.000  52  114
3.000  54  146
3.810  51  226
4.000  51  236
5.000  50  296
6.000  49  360
7.000  50  416
8.000  51  484
9.000  48  540
10.00  47  596
""")
df = pd.read_csv(tb, delim_whitespace=True)

print(df)

# plt.plot(df.V_pulser, df.RT_stage2_ns, '-r')
# plt.xlabel('V_pulser', ha='right', x=1)
# plt.ylabel('rise time (ns)', ha='right', y=1)
# plt.gca().tick_params('y', colors='r')
# 
# p1a = plt.gca().twinx()
# p1a.plot(df.V_pulser, df.mVpp_stage2, '-b')
# p1a.set_ylabel('Pulse height (mVpp)', color='b', ha='right', y=1)
# p1a.tick_params('y', colors='b')

plt.plot(df.V_pulser, df.V_pulser/df.RT_stage2_ns, '-r')
plt.xlabel('V_pulser', ha='right', x=1)
plt.ylabel('Slew Rate (mV/ns)', ha='right', y=1)

# plt.show()
plt.savefig('./plots/slew_rate.png', dpi=150)