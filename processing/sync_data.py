#!/usr/bin/env python3
import os
import glob
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import subprocess as sp
from pprint import pprint
from pygama.utils import *

def main():
    """
    sync CAGE data with NERSC.
    """
    with open("cage.json") as f:
        expDB = json.load(f)

    # run_rsync(expDB)
    daq_cleanup(expDB)


def run_rsync(expDB, test=False):
    """
    TODO: update w/ globus + cron
    NOTE: this doesn't work yet for wisecg, file permissions
    at the source directory appear to be messed up.  Need to
    talk with Robert Varner to fix.
    """
    if "mjcenpa" not in os.environ["USER"]:
        print("Error, we're not on the MJ60 DAQ machine.  Exiting ...")
        exit()

    daq_dir = os.path.expandvars(expDB["daq_dir"] + "/")
    daq_nersc = "{}:{}/".format(expDB["nersc_login"], expDB["nersc_dir"])

    if test:
        cmd = "rsync -avh --dry-run {} {}".format(daq_dir, daq_nersc)
    else:
        # cmd = "rsync -avh --no-perms {} {}".format(daq_dir, daq_nersc)
        # try to fix permissions issue?
        cmd = "rsync -avh --no-o --no-g --no-perms {} {}".format(daq_dir, daq_nersc)
    sh(cmd)


def daq_cleanup(expDB):
    """
    build a list of files on the DAQ and nersc, check integrity,
    and delete files on the DAQ only if we're sure the transfer was successful.

    # done via subprocess because we have to do it that way for NERSC
    # awk note: col 9: filename, col 5: filesize

    # locally, we need: file name, full path to DAQ file, size in bytes
    # remotely, we need: file name, size in bytes

    need to account file sizes that don't match.  Also at NERSC, file sizes
    can be slightly different than on local machines (something about
    accounting for a lot of files)
    """
    if "mjcenpa" not in os.environ["USER"]:
        print("Error, we're not on the MJ60 DAQ machine.  Exiting ...")
        exit()

    nersc_login = 'grsong@cori.nersc.gov'

    print('Building local and remote file lists ...')

    # local (DAQ) list
    datadir_loc = os.path.expandvars(expDB["daq_dir"] + "/")
    args = f"du -a {datadir_loc} | awk '{{print $1, $2}}'"
    cols = sp.check_output(args, shell=True)
    cols = cols.decode('utf-8')
    cols = cols.split('\n')

    # make a dataframe keyed by file name
    files_loc = {}
    for row in cols:
        tmp = row.split(' ')
        if len(tmp) < 2: continue
        fname = tmp[1].split('/')[-1]
        fpath = '/'.join(t for t in tmp[1].split('/')[:-1])
        files_loc[fname] = {'size':int(tmp[0]), 'path':fpath}
        # print(fname, files_loc[fname])
    df_local = pd.DataFrame.from_dict(files_loc, orient='index')
    # print(df_local.to_string())

    # remote (NERSC) list
    args = f"ssh {nersc_login} du -a {expDB['nersc_dir']} | awk '{{print $1, $2}}'"
    cols = sp.check_output(args, shell=True)
    cols = cols.decode('utf-8')
    cols = cols.split('\n')

    files_nersc = {}
    for row in cols:
        tmp = row.split(' ')
        if len(tmp) < 2: continue
        fname = tmp[1].split('/')[-1]
        fpath = '/'.join(t for t in tmp[1].split('/')[:-1])
        files_nersc[fname] = {'size':int(tmp[0]), 'path':fpath}
        # print(fname, files_nersc[fname])

    df_remote = pd.DataFrame.from_dict(files_nersc, orient='index')
    # print(df_remote)

    # look at files where we found a match in the remote list
    idx_found = df_local.index.intersection(df_remote.index)
    # print(df_local.loc[idx_found])
    # print(df_remote.loc[idx_found])

    # concat the dfs and create a list of local files to delete
    df_found = df_local.loc[idx_found]
    df_found['remote_size'] = df_remote.loc[idx_found]['size']
    df_found['remote_path'] = df_remote.loc[idx_found]['path']

    # ignore directories, certain regex's, and check file sizes
    def mark_for_deletion(row):
        fname = row['path'] + '/' + row.name
        if os.path.isdir(fname): return False

        ignore_list = [".Orca", "RunNumber", "openFiles"]
        if any(ig in fname for ig in ignore_list): return False

        # compare local and remote file sizes
        # there is actually a huge variation in this number, because of
        # the weird nersc filesystem.  But what we CAN do, is make sure
        # the ratio isn't less than 1 -- this seems to always be true
        if row['remote_size'] == 0:
            print(f'Warning, file may be incompletely backed up:\n  {fname}'
                  f"\n loc_size: {row['size']}  remote size: {row['remote_size']}")
            return False
        if row['size'] / row['remote_size'] < 1:
            print(f'Warning, file may be incompletely backed up:\n  {fname}'
                  f"\n loc_size: {row['size']}  remote size: {row['remote_size']}")
            return False

        # ok, the file has been backed up
        return True

    # get files to delete
    df_found['deleteme'] = df_found.apply(mark_for_deletion, axis=1)
    df_del = df_found.loc[df_found.deleteme]
    files_del = (df_del['path'] + '/' + df_del.index).values
    # print(files_del)

    # show files without a match in the remote list (not backed up)
    idx_missing = df_local.index.difference(df_local.loc[idx_found].index)
    df_missing = df_local.loc[idx_missing]
    if len(df_missing) > 0:
        print('Found files on DAQ not backed up to NERSC, these will NOT be deleted:')
        print(df_missing.to_string())

    print(f'\nSuccessfully backed up {len(df_del)} files to NERSC:')
    print(df_del[['size','path']].to_string())

    # now delete old files, ask for Y/N confirmation
    now = datetime.now()
    ans = input("OK to delete backed up files? y/n:")
    if ans.lower() == 'y':
        for f in tqdm(files_del):
            os.remove(f)
        print("DAQ backup & purge completed!")
    else:
        print("Files not deleted.")
    print(datetime.now().strftime("%Y-%m-%d %H:%M"))


if __name__=="__main__":
    main()
