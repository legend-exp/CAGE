set -x
date
source /opt/geant4/share/Geant4-10.5.1/geant4make/geant4make.sh
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/root/lib:/opt/hdf5/lib"
DATADIR="${USERDIR}/jobs/data/${JOB_NAME}"
mkdir -p ${DATADIR}
cd $CAGE_SW/new_sims
g4simple ./macros/oppi/scan/y5_thetaDet90_rotary145_241Am_100000000.mac 
cd $TMPDIR
h5repack -v -f GZIP=5 .hdf5 ${DATADIR}/g4simpleout_${SGE_TASK_ID}.hdf5
date
set +x
