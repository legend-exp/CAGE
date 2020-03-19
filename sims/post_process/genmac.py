import sys
file = open('/Users/mattstortini/Desktop/run.mac.in')
contents = file.read()
replaced_contents = contents.replace('JOBID.out', sys.argv[1])
print(replaced_contents)
