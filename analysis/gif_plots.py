#!/usr/bin/env python3
import imageio
import glob


def main():

    runs = [64, 66, 70, 72, 60] #starting at 7.5mm so if doesn't show up in ppt at least will show a plot with alphas
    # plot_dir = './plots/normScan/cal_normScan/'

    # plot_dir = './plots/angleScan/'
    plot_dir = './plots/new_normScan/'

    # radii = ['10', '15', '18'] #source angle scan
    radii = ['7_5', '12_5', '17_5', '22_5', '2_5'] #norm scan
    
    # dsp_params = ['alp_ToE_60keV_']
    # outputs_files = ['toeVsE_60keV.gif']
    
    # dsp_params = ['alp_ToEvDCR_', 'alp_ToE_60keV_', 'alp_ToE_']
    # outputs_files = ['toeVsdcr.gif','toeVsE_60keV.gif', 'toeVsE.gif']

    dsp_params = ['alp_energy_60keV_', 'alp_energy_', 'alp_AoE_', 'alp_DCR_', 'alp_AoEvDCR_', 'alp_DCRvTp0220_', 'alp_ToE_60keV_']
    outputs_files = ['energy_60keV.gif','energy.gif', 'aoeVsE.gif', 'dcrVsE.gif', 'aoeVsdcr.gif', 'dcrVstp0220.gif', 'toeVsE_60keV.gif']

    dsp_gif(dsp_params, runs, radii, plot_dir, outputs_files)
    # dsp_gif_byRadius(dsp_params, radii, plot_dir, outputs_files)

def dsp_gif(dsp_params, runs, radii, plot_dir, outputs_files):

    for param, outfile in zip(dsp_params, outputs_files):
        images = []
        for run, radius in zip(runs, radii):
            images.append(imageio.imread(f'{plot_dir}normalized_{param}{radius}mm_90deg_run{run}.png'))

        imageio.mimsave(f'{plot_dir}{outfile}', images, fps=1)
        print('Saving ', f'{plot_dir}{outfile}')

def dsp_gif_byRadius(dsp_params, radii, plot_dir, outputs_files):

    for param, outfile in zip(dsp_params, outputs_files):
        for radius in radii:
            images = []
            for file in sorted(glob.glob(f'{plot_dir}normalized_{param}{radius}*.png'), reverse=True):
                print(radius, file)
                images.append(imageio.imread(file))

            imageio.mimsave(f'{plot_dir}normalized_{radius}mm_{outfile}', images, fps=1)
            print('Saving ', f'{plot_dir}normalized_{radius}mm_{outfile}')


if __name__=="__main__":
    main()
