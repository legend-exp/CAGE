### optimization parameter breakdown
Clint, July 2021, v6 electronics

Common wf windows:
```
0:3500     baseline
4250:8000  pole zero for DCR
4250:5500  alternate mean_stdev, for fltp2 & fltp2_sig
```

Priority to optimize:
1. wf_pz & wf_pzDCR
2. wf_trap & trapEftp
3. dcr_trap & dcr

```
name --      -- function --      -- default inputs --

wf_trap      trap_norm           wf_pz, 1*us, 4*us
wf_atrap     asymTrapFilter      20*ns, 1*us, 4*us, wf_pz
trapEftp     fixed_time_pickoff  tp_0 + (1*us+3*us), wf_trap
wf_pz        double_pole_zero    51.00*us, 4*us, 0.05
fltp2, fltp2_sig
             mean_stdev          wf_pz[4250:5500]
wf_pzDCR     double_pole_zero    51.00*us, 4*us, 0.05
fltpDCR, fltpDCR_sig
             mean_stdev          wf_pzDCR[4250:8000]
dcr_trap     trap_norm           wf_pzDCR, 750, 2250
dcr          fixed_time_pickoff  79*us

# lower priority:

bl_trap      trap_norm           500, 500
bl_slope_ftp fixed_time_pickoff  7.5*us
bl, bl_sig   mean_stdev          waveform[0:3500]
bl_mean, bl_sig, bl_slope, bl_int
             linear_slope_fit    waveform[0:3500]
ct_corr      trap_pickoff        wf_pz, 1.5*us, 0
log_tail     np.log              wf_blsub[4250:8000]
curr         avg_current         10
wf_triangle  trap_norm           wf_pz, 100*ns, 10*ns
wf_psd       psd                 wf_blsub[:3000]
hf_max       np.amax             10*mhz*3000, 20*mhz*3000
lf_max       np.amax             0, 150*khz*3000
```
