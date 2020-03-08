import numpy as np
import h5py
import pandas as pd
import sys
import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

def main():

    weighted_hits()
    #initial_hits()
    #incident_KE()

def weighted_hits():

    if(len(sys.argv) != 2):
        print('Usage: x_hits.py [Source]')
        sys.exit()

    df =  pd.read_hdf("processed.hdf5", key="procdf")
    df['energy'] = df['energy']*1000
    df_01 = df.loc[(df.energy>0)&(df.energy<100)]
    df_02 = df.loc[(df.energy>100)&(df.energy<1000)]
    df_03 = df.loc[(df.energy>1000)&(df.energy<1500)]
    df_04 = df.loc[(df.energy>1500)&(df.energy<2000)]
    df_05 = df.loc[(df.energy>2000)]

    m = np.array(df['x'])
    m_01 = np.array(df_01['x'])
    m_02 = np.array(df_02['x'])
    m_03 = np.array(df_03['x'])
    m_04 = np.array(df_04['x'])
    m_05 = np.array(df_05['x'])

    plt.hist(m, np.arange(-32,32,0.5), histtype='step', color='black', label='all energies, {} entries'.format(len(df)))
    plt.hist(m_01, np.arange(-32,32,0.5), histtype='step', color='red', label='0<E<100 keV, {} entries'.format(len(m_01)))
    plt.hist(m_02, np.arange(-32,32,0.5), histtype='step', color='green', label='100<E<1000 keV, {} entries'.format(len(m_02)))
    plt.hist(m_03, np.arange(-32,32,0.5), histtype='step', color='purple', label='1000<E<1500 keV, {} entries'.format(len(m_03)))
    plt.hist(m_04, np.arange(-32,32,0.5), histtype='step', color='aqua', label='1500<E<2000 keV, {} entries'.format(len(m_04)))
    plt.hist(m_05, np.arange(-32,32,0.5), histtype='step', color='navy', label='E>2000 keV, {} entries'.format(len(m_05)))

    plt.xlim(-32,32)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('x position (mm)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Weighted Position Measurements ('+sys.argv[1]+' Source)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

def initial_hits():

    if(len(sys.argv) != 2):
        print('Usage: x_hits.py [Source]')
        sys.exit()

    df =  pd.read_hdf("processed_initial_hits.hdf5", key="procdf")
    #df = df.loc[(df.KE>0)]
    df_electrons = df.loc[(df.pid==11)]
    df_gammas = df.loc[(df.pid==22)]
    df_alphas = df.loc[(df.pid==1000020040)]

    m_all = np.array(df['x'])
    m_electrons = np.array(df_electrons['x'])
    m_gammas = np.array(df_gammas['x'])
    m_alphas = np.array(df_alphas['x'])

    plt.hist(m_all, np.arange(-32,32,0.5), histtype='step', color='black', label='total hits, {} entries'.format(len(m_all)))
    plt.hist(m_electrons, np.arange(-32,32,0.5), histtype='step', color='red', label='electron hits, {} entries'.format(len(m_electrons)))
    plt.hist(m_gammas, np.arange(-32,32,0.5), histtype='step', color='green', label='gamma hits, {} entries'.format(len(m_gammas)))
    plt.hist(m_alphas, np.arange(-32,32,0.5), histtype='step', color='purple', label='alpha hits, {} entries'.format(len(m_alphas)))

    plt.xlim(-32,32)
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('x position (mm)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Initial Detector Hits by Particle Type ('+sys.argv[1]+' Source)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.show()

def incident_KE():

    if(len(sys.argv) != 3):
        print('Usage: spectrum.py [maximum energy value for x axis of plot in keV] [Source]')
        sys.exit()

    df =  pd.read_hdf("processed_initial_hits.hdf5", key="procdf")
    df['incident_KE'] = df['KE']+df['Edep']
    df_electrons = df.loc[(df.pid==11)&(df.incident_KE>0)]
    df_gammas = df.loc[(df.pid==22)&(df.incident_KE>0)]
    df_alphas = df.loc[(df.pid==1000020040)&(df.incident_KE>0)]

    ke_electrons = list(df_electrons['incident_KE'])
    ke_electrons = list(x*1000 for x in ke_electrons)
    ke_gammas = list(df_gammas['incident_KE'])
    ke_gammas = list(x*1000 for x in ke_gammas)
    ke_alphas = list(df_alphas['incident_KE'])
    ke_alphas = list(x*1000 for x in ke_alphas)

    plt.hist(ke_electrons, np.arange(0,int(sys.argv[1]),1), histtype='step', color = 'red', label='electron hits, {} entries'.format(len(ke_electrons)))
    plt.hist(ke_gammas, np.arange(0,int(sys.argv[1]),1), histtype='step', color = 'green', label='gamma hits, {} entries'.format(len(ke_gammas)))
    plt.hist(ke_alphas, np.arange(0,int(sys.argv[1]),1), histtype='step', color = 'purple', label='alpha hits, {} entries'.format(len(ke_alphas)))
    plt.xlim(0,int(sys.argv[1]))
    plt.ylim(0,plt.ylim()[1])
    plt.xlabel('Kinetic Energy (keV)', ha='right', x=1.0)
    plt.ylabel('Counts', ha='right', y=1.0)
    plt.title('Incident KE by Particle Type ('+sys.argv[2]+' Source)')
    plt.legend(frameon=True, loc='upper right', fontsize='small')
    plt.tight_layout()
    #plt.semilogy()
    #plt.semilogx()
    plt.show()

if __name__ == '__main__':
        main()
