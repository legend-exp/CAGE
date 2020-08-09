import numpy as np
import h5py
import pandas as pd
import sys
import time
from numba import jit
from tqdm import tqdm
#np.set_printoptions(threshold=sys.maxsize)

def main():

    #process_with_dead_layer()
    process()
    #process_initial_hits()
    #process_muon_in_veto()
    #process_muon_without_veto()
    #process_muon_with_veto()

def process_with_dead_layer():

    if(len(sys.argv) != 2):
        print('Usage: postprocesshdf5.py [filename.hdf5]')
        sys.exit()

    start = time.time()
    print('In Progress...')

    pd.options.mode.chained_assignment = None

    # the following four functions define the dead layer (outer edge) and the passivated surface.
    # these functions are never called because numba doesn't work with previously defined functions ...
    # but in the numba for loop they are used in the order they are defined here
    def expo_DL(r, a, b, sigma):
        return 1-(np.exp(sigma*(a-r))/(-1+np.exp(sigma*a)-a*sigma+b*sigma)+(1+a*sigma-b*sigma)/(1-np.exp(sigma*a)+a*sigma-b*sigma))

    def linear_DL(r, a, b, sigma):
        return 1-(sigma*r/(1-np.exp(a*sigma)+a*sigma-b*sigma)+b*sigma/(-1+np.exp(a*sigma)-a*sigma+b*sigma))

    def expo_PS(z, a_ps, b_ps, sigma_ps):
        return 1-(np.exp(sigma_ps*(a_ps-z))/(-1+np.exp(sigma_ps*a_ps)-a_ps*sigma_ps+b_ps*sigma_ps)+(1+a_ps*sigma_ps-b_ps*sigma_ps)/(1-np.exp(sigma_ps*a_ps)+a_ps*sigma_ps-b_ps*sigma_ps))

    def linear_PS(z, a_ps, b_ps, sigma_ps):
        return 1-(sigma_ps*z/(1-np.exp(a_ps*sigma_ps)+a_ps*sigma_ps-b_ps*sigma_ps)+b_ps*sigma_ps/(-1+np.exp(a_ps*sigma_ps)-a_ps*sigma_ps+b_ps*sigma_ps))

    # constants that define dead layer and passivated surface
    a = np.float64(0.7)
    b = np.float64(1)
    sigma = np.float64(-2)
    a_ps = np.float64(0.0006)
    b_ps = np.float64(0.001)
    sigma_ps = np.float64(-3000)
    pctResAt1MeV = .15
    p = [0.2121, 0.01838, 0.00031137]

    # import g4simple output file, organize the data into a pandas dataframe, and keep events that happen in the detector
    g4sfile = h5py.File(sys.argv[1], 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']
    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                       columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),
                       columns=['x']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),
                       columns=['y']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),
                       columns=['z']), lsuffix = '_caller', rsuffix = '_other')

    detector_hits = g4sdf.loc[(g4sdf.volID==1)]

    # add in Penetration Length to dataframe, and transform relevant columns of dataframes into numpy arrays to work with numba
    detector_hits['r_p'] = 31 + 3.553e-15 - np.sqrt(detector_hits['x']**2+detector_hits['y']**2)
    detector_hits['z_p'] = 23 - detector_hits['z']
    r_p = detector_hits['r_p'].values
    z_p = detector_hits['z_p'].values
    Edep = detector_hits['Edep'].values
    activeness = np.zeros(len(detector_hits), dtype=np.float64)

    # for loop to give the activeness at the location of an energy deposition
    @jit(nopython=True)
    def activeness_loop(r,z,activeness):
        for i in range(len(activeness)):
            if 0<=r[i]<=a and b_ps<=z[i]:
                activeness[i] = 1-(np.exp(sigma*(a-r[i]))/(-1+np.exp(sigma*a)-a*sigma+b*sigma)+(1+a*sigma-b*sigma)/(1-np.exp(sigma*a)+a*sigma-b*sigma))
            elif a<r[i]<=b and b_ps<=z[i]:
                activeness[i] = 1-(sigma*r[i]/(1-np.exp(a*sigma)+a*sigma-b*sigma)+b*sigma/(-1+np.exp(a*sigma)-a*sigma+b*sigma))
            elif 0<=z[i]<=a_ps:
                activeness[i] = 1-(np.exp(sigma_ps*(a_ps-z[i]))/(-1+np.exp(sigma_ps*a_ps)-a_ps*sigma_ps+b_ps*sigma_ps)+(1+a_ps*sigma_ps-b_ps*sigma_ps)/(1-np.exp(sigma_ps*a_ps)+a_ps*sigma_ps-b_ps*sigma_ps))
            elif a_ps<z[i]<=b_ps:
                activeness[i] = 1-(sigma_ps*z[i]/(1-np.exp(a_ps*sigma_ps)+a_ps*sigma_ps-b_ps*sigma_ps)+b_ps*sigma_ps/(-1+np.exp(a_ps*sigma_ps)-a_ps*sigma_ps+b_ps*sigma_ps))
            elif b<r[i] and b_ps<z[i]:
                activeness[i] = 1
        return activeness

    activeness_loop(r_p, z_p, activeness)

    # transform energy depositions into actual energy collected
    E_collected = activeness*Edep
    detector_hits['E_collected'] = E_collected

    # sum energy collections for each event, and apply energy resolution function
    procdf = pd.DataFrame(detector_hits.groupby(['event','volID'], as_index=False)['E_collected'].sum())
    procdf = procdf.rename(columns={'E_collected':'energy'})
    procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*np.random.randn(len(procdf['energy']))*pctResAt1MeV/100

    # remove unnecessary columns, and save the processed dataframe
    del procdf['event']
    del procdf['volID']
    procdf = procdf.reset_index(drop=True)

    procdf.to_hdf('processed_with_dead_layer.hdf5', key='procdf', mode='w')

    end = time.time()
    print('Done --- processed_with_dead_layer.hdf5 has been output!')
    print('Run time = {:.0f} seconds'.format(end-start))

def process():

    if(len(sys.argv) != 2):
        print('Usage: postprocesshdf5.py [filename.hdf5]')
        sys.exit()

    start = time.time()
    print('In Progress...')

    pd.options.mode.chained_assignment = None

    pctResAt1MeV = .15 #percent resolution at 1 MeV
    p = [0.2121, 0.01838, 0.00031137] # E-res fit fuction parameters

    g4sfile = h5py.File(sys.argv[1], 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']

    #n=list(g4sntuple.keys())
    #print(n)

    ##taking data from g4sntuple and organizing it into a frame. the groups for which there is data is given by print(n).
    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                       columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['pid']['pages']),
                       columns=['pid']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),
                       columns=['x']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),
                       columns=['y']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),
                       columns=['z']), lsuffix = '_caller', rsuffix = '_other')


    detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)] #energy + phys volume cut
    detector_hits['x_weights'] = detector_hits['x'] * detector_hits['Edep']
    detector_hits['y_weights'] = detector_hits['y'] * detector_hits['Edep']
    detector_hits['z_weights'] = detector_hits['z'] * detector_hits['Edep']

    procdf= pd.DataFrame(detector_hits.groupby(['event','volID'], as_index=False)['Edep','x_weights','y_weights', 'z_weights'].sum())
    # rename the summed energy depositions for each step within the event to "energy". This is analogous to the event energy you'd see in your detector

    procdf = procdf.rename(columns={'Edep':'energy'})

    procdf['x'] = procdf['x_weights']/procdf['energy']
    procdf['y'] = procdf['y_weights']/procdf['energy']
    procdf['z'] = procdf['z_weights']/procdf['energy']

    del procdf['x_weights']
    del procdf['y_weights']
    del procdf['z_weights']

    ##apply energy resolution function
    procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*np.random.randn(len(procdf['energy']))*pctResAt1MeV/100
    #procdf['Edep'] = procdf['Edep'] + np.sqrt(p[0]**2+p[1]**2*procdf['Edep']+p[2]**2*procdf['Edep']**2)*np.random.randn(len(procdf['Edep']))*pctResAt1MeV

    procdf.to_hdf('processed.hdf5', key='procdf', mode='w')

    end = time.time()
    print('Done --- processed.hdf5 has been output!')
    print('Run time = {:.0f} seconds'.format(end-start))

def process_initial_hits():

    if(len(sys.argv) != 2):
        print('Usage: postprocesshdf5.py [filename.hdf5]')
        sys.exit()

    start = time.time()
    print('In Progress...')

    pd.options.mode.chained_assignment = None

    pctResAt1MeV = .15
    p = [0.2121, 0.01838, 0.00031137]

    g4sfile = h5py.File(sys.argv[1], 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']

    ##taking data from g4sntuple and organizing it into a frame. the groups for which there is data is given by print(n).
    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['KE']['pages']), columns=['KE']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                       columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['pid']['pages']),
                       columns=['pid']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),
                       columns=['x']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),
                       columns=['y']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),
                       columns=['z']), lsuffix = '_caller', rsuffix = '_other')

    df = g4sdf.loc[(g4sdf.volID==1)]

    event = df['event'].values
    nothing = np.zeros(len(df), dtype=np.float64)

    @jit(nopython=True)
    def initial_hits_loop(array1, array_cutter):
        for i in range(0, len(array_cutter)-1):
            a = int(i)+1
            if array1[i]==array1[a]:
                array_cutter[a] = 1
        return array_cutter

    initial_hits_loop(event, nothing)

    df['nothing'] = nothing

    procdf = df.loc[(df.nothing<1)&(df.volID==1)]
    del procdf['nothing']

    procdf.to_hdf('processed_initial_hits.hdf5', key='procdf', mode='w')

    end = time.time()
    print('Done --- processed_initial_hits.hdf5 has been output!')
    print('Run time = {:.0f} seconds'.format(end-start))

def process_muon_in_veto():

    if(len(sys.argv) != 3):
        print('Usage: postprocesshdf5.py [filename.hdf5] [veto number]')
        sys.exit()

    start = time.time()
    print('In Progress...')

    pd.options.mode.chained_assignment = None

    pctResAt1MeV = .15
    p = [0.2121, 0.01838, 0.00031137]

    g4sfile = h5py.File(sys.argv[1], 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']

    ##taking data from g4sntuple and organizing it into a frame. the groups for which there is data is given by print(n).
    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                       columns=['volID']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['pid']['pages']),
                       columns=['pid']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['x']['pages']),
                       columns=['x']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['y']['pages']),
                       columns=['y']), lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['z']['pages']),
                       columns=['z']), lsuffix = '_caller', rsuffix = '_other')

    ##apply E cut / detID cut
    veto_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==2)]

    ##sum energy collections for each event.
    procdf = pd.DataFrame(veto_hits.groupby(['event','volID'], as_index=False)['Edep'].sum())
    procdf = procdf.rename(columns={'Edep':'energy'})

    ##apply energy resolution function
    #procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*np.random.randn(len(procdf['energy']))*pctResAt1MeV/100
    #procdf['Edep'] = procdf['Edep'] + np.sqrt(p[0]**2+p[1]**2*procdf['Edep']+p[2]**2*procdf['Edep']**2)*np.random.randn(len(procdf['Edep']))*pctResAt1MeV

    procdf.to_hdf('muon_processed_in_veto_'+sys.argv[2]+'.hdf5', key='procdf', mode='w')

    end = time.time()
    print('Done --- muon_processed_in_veto_'+sys.argv[2]+'.hdf5 has been output!')
    print('Run time = {:.0f} seconds'.format(end-start))

def process_muon_without_veto():

    if(len(sys.argv) != 3):
        print('Usage: postprocesshdf5.py [filename.hdf5] [veto number]')
        sys.exit()

    start = time.time()
    print('In Progress...')

    pd.options.mode.chained_assignment = None

    pctResAt1MeV = .15
    p = [0.2121, 0.01838, 0.00031137]

    g4sfile = h5py.File(sys.argv[1], 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']

    #n=list(g4sntuple.keys())
    #print(n)

    ##taking data from g4sntuple and organizing it into a frame. the groups for which there is data is given by print(n).
    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                       columns=['volID']), lsuffix = '_caller', rsuffix = '_other')

    ##to see this dataframe simply print(g4sdf).

    ##apply E cut / detID cut
    detector_hits = g4sdf.loc[(g4sdf.Edep>0)&(g4sdf.volID==1)]

    ##sum energy collections for each event.
    procdf= pd.DataFrame(detector_hits.groupby(['event','volID'], as_index=False)['Edep'].sum())

    Edep = procdf['Edep'].values
    energy = np.zeros(len(Edep), dtype=np.float64)

    @jit(nopython=True)
    def muon_overflow(array1, array_out):
        for i in range(len(array1)):
            if array1[i] > 3.15:
                array_out[i] = 3.15
            else:
                array_out[i] = array1[i]
        return array_out

    muon_overflow(Edep, energy)

    procdf['energy'] = energy

    ##apply energy resolution function
    #procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*np.random.randn(len(procdf['energy']))*pctResAt1MeV/100
    #procdf['Edep'] = procdf['Edep'] + np.sqrt(p[0]**2+p[1]**2*procdf['Edep']+p[2]**2*procdf['Edep']**2)*np.random.randn(len(procdf['Edep']))*pctResAt1MeV

    procdf.to_hdf('muon_processed_without_veto_'+sys.argv[2]+'.hdf5', key='procdf', mode='w')

    end = time.time()
    print('Done --- muon_processed_without_veto_'+sys.argv[2]+'.hdf5 has been output!')
    print('Run time = {:.0f} seconds'.format(end-start))

def process_muon_with_veto():

    if(len(sys.argv) != 3):
        print('Usage: postprocesshdf5.py [filename.hdf5] [veto number]')
        sys.exit()

    start = time.time()
    print('In Progress...')

    pd.options.mode.chained_assignment = None

    pctResAt1MeV = .15
    p = [0.2121, 0.01838, 0.00031137]

    g4sfile = h5py.File(sys.argv[1], 'r')
    g4sntuple = g4sfile['default_ntuples']['g4sntuple']

    #n=list(g4sntuple.keys())
    #print(n)

    ##taking data from g4sntuple and organizing it into a frame. the groups for which there is data is given by print(n).
    g4sdf = pd.DataFrame(np.array(g4sntuple['event']['pages']), columns=['event'])
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['step']['pages']), columns=['step']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['Edep']['pages']), columns=['Edep']),
                       lsuffix = '_caller', rsuffix = '_other')
    g4sdf = g4sdf.join(pd.DataFrame(np.array(g4sntuple['volID']['pages']),
                       columns=['volID']), lsuffix = '_caller', rsuffix = '_other')

    ##to see this dataframe simply print(g4sdf).

    ##apply E cut / detID cut
    E_Cut = g4sdf.loc[(g4sdf.Edep>0)]

    ##sum energy collections for each event.
    procdf= pd.DataFrame(E_Cut.groupby(['event','volID'], as_index=False)['Edep'].sum())

    Edep = procdf['Edep'].values
    volID = procdf['volID'].values
    event = procdf['event'].values

    @jit(nopython=True)
    def veto(volumes_array,events_array,energy_array):
        for i in range(1, len(energy_array)-1):
            a = int(i)-1
            if volumes_array[i]==2 and energy_array[i] > 3 and volumes_array[a]==1 and events_array[i]==events_array[a]:
                energy_array[a]=0
        return energy_array

    veto(volID,event,Edep)

    procdf['Edep'] = Edep

    procdf = procdf.loc[(procdf.Edep>0)&(procdf.volID==1)]

    Edep = procdf['Edep'].values
    energy = np.zeros(len(Edep), dtype=np.float64)

    @jit(nopython=True)
    def muon_overflow(array1, array_out):
        for i in range(len(array1)):
            if array1[i] > 3.15:
                array_out[i] = 3.15
            else:
                array_out[i] = array1[i]
        return array_out

    muon_overflow(Edep, energy)

    procdf['energy'] = energy

    ##apply energy resolution function
    #procdf['energy'] = procdf['energy'] + np.sqrt(procdf['energy'])*np.random.randn(len(procdf['energy']))*pctResAt1MeV/100
    #procdf['Edep'] = procdf['Edep'] + np.sqrt(p[0]**2+p[1]**2*procdf['Edep']+p[2]**2*procdf['Edep']**2)*np.random.randn(len(procdf['Edep']))*pctResAt1MeV

    procdf.to_hdf('muon_processed_with_veto_'+sys.argv[2]+'.hdf5', key='procdf', mode='w')

    end = time.time()
    print('Done --- muon_processed_with_veto_'+sys.argv[2]+'.hdf5 has been output!')
    print('Run time = {:.0f} seconds'.format(end-start))

if __name__ == '__main__':
        main()
