export EDITOR=/usr/bin/vi

# alias root="root -l"
#alias root="root -l $GATDIR/LoadGATClasses.C"
#alias vi="vi -S ~/.virc"
alias ls="ls -G"

export PATH="$(brew --prefix)/opt/python3/bin:${PATH}"
alias python=python3
alias ipy=ipython
alias pip=pip3

alias rocks='ssh cenpa-rocks.npl.washington.edu -lwisecg'

# for rmate
export PATH=${HOME}:${PATH}
function rpf {
    # kill hanging remote port forwarding processes (useful for rmate)
    myarr=$(ps -u `whoami` | grep sshd | awk '{print $2}')
    kill $myarr
}

# for gclib (newmark motors)
export DYLD_LIBRARY_PATH="/Applications/gclib/dylib/:${DYLD_LIBRARY_PATH}"

# ipython
export PATH=${HOME}/Library/Python/3.7/bin:${PATH}

export DATADIR=~/Data
export MJSWDIR=~/Dev
export CXXFLAGS='-g -std=c++11'

# export ROOTSYS=$MJSWDIR/root-6.12.06/install
export ROOTSYS=$MJSWDIR/ROOT/build
source $ROOTSYS/bin/thisroot.sh

export CLHEP_BASE_DIR="$MJSWDIR/CLHEP/2.4.0.1-build"
export PATH="$CLHEP_BASE_DIR:$PATH"
export DYLD_LIBRARY_PATH="$CLHEP_BASE_DIR/lib:$DYLD_LIBRARY_PATH"

export MGDODIR="$MJSWDIR/MGDO"
export PATH="$MGDODIR/install/bin:$PATH"
export DYLD_LIBRARY_PATH="$MGDODIR/lib:$MGDODIR/install/lib:$DYLD_LIBRARY_PATH"

export TAMDIR="$MGDODIR/tam"
export DYLD_LIBRARY_PATH="$TAMDIR/lib:$DYLD_LIBRARY_PATH"

export ORDIR="$MJSWDIR/OrcaRoot"
export PATH="$ORDIR/Applications:$PATH"
export DYLD_LIBRARY_PATH="$ORDIR/lib:$DYLD_LIBRARY_PATH"

export MJORDIR="$MJSWDIR/MJOR"
export PATH="$MJORDIR:$PATH"
export DYLD_LIBRARY_PATH="$MJORDIR:$DYLD_LIBRARY_PATH"

export GATDIR="$MJSWDIR/GAT"
export PATH="$GATDIR/Scripts:$GATDIR/Apps:$PATH"
export DYLD_LIBRARY_PATH="$GATDIR/lib:$DYLD_LIBRARY_PATH"

export ROOT_INCLUDE_PATH="$GATDIR/BaseClasses:$GATDIR/MGTEventProcessing:$GATDIR/MGOutputMCRunProcessing:$MGDODIR/Base:$MGDODIR/Gerda:$MGDODIR/GerdaTransforms:$MGDODIR/Majorana:$MGDODIR/MJDB:$MGDODIR/Root:$MGDODIR/Tabree:$MGDODIR/Tools:$MGDODIR/Transforms:$TAMDIR:$MGDODIR/install/include/mgdo:$MGDODIR/install/include/tam"

export MYDATADIR="~/Data"

pyopenh5() {
  echo "opening hdf5 file $1 as f..."
  python -i -c "import h5py; f = h5py.File('$1', 'r')"
}
