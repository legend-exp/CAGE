import sys
N = 10
if len(sys.argv) > 1:
    N = int(sys.argv[1])
for n in range(N):
    print('echo qsub mcjob.csh ' + str(n + 1))
