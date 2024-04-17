import mdtraj as md

timestep=0.005
steps=1000000000 # Simulation steps
interval=5000 # steps between dcd dumps
#dump_to_frame_ratio=2.5 # How many temp, etc, readouts per dcd frame save? 
temp=3 # Via the ordering of them. e.g. .7 .8 .9 1.0 1.1 1.2 1.3 would use temp=3 if T=1.0 is the temp of interest
dump_files=[]
for k in ["0.7","0.8","0.9","1.0","1.1","1.2","1.3"]: # Should be in the order at which defined in lammps in file
    dump_files.append("DUMP_FILE_temp{}.dcd".format(k))

#######
# Analyze log.lammps file functions

def to_space(X):
    # Find the first space (if exists) in a string. 
    # Return characters before it
    n=0
    while n < len(X) and X[n] != ' ':
        n+=1
    if n < len(X)-1:
        return X[0:n],X[n+1:]
    else:
        return X[0:n],""

def remove_bad_lines(X):
    # Find all lines that don't start with an integer
    # (i.e. that don't contain swap info) and remove them.
    # Takes array filled with strings, each string a line
    # from a file

    X2=[]
    for k in range(len(X)):
        x,_=to_space(X[k])
        try:
            int(x)
            if X[k][-1:] == '\n':
                X2.append(X[k][0:-1])
            else:
                X2.append(X[k])
        except:
            pass
    return X2   

def get_info(X):
    # Get the frames, proper replica for the center temperature in each case
    frames=[]
    replica=[]
    for K in X:
        frame,K=to_space(K)
        for j in range(temp):
            _,K=to_space(K)
        K,_=to_space(K)  
        frames.append(int(frame))
        replica.append(int(K))
    return frames,replica

######
# Function for getting time information from dump file log files
def get_times(X):
    # Take the time information from the log files and return as an array
    n=0
    times=[]
    for k in range(len(X)):
        K=X[k]
        K=K.lstrip()
        time,_=to_space(K)
        if n == 2:
            try:
                times.append(float(time))
                continue
            except: # Made it to the end of the lines corresponding to dump info
                break
        if time == "Time": # Reached the line _before_ the dump info lines
            n+=1
    return times


######
# Prepare the information from log.lammps file
log_file=open("log.lammps",'r')
log_file=log_file.readlines()
log_file=remove_bad_lines(log_file)

frames,replica=get_info(log_file)

##### 
# Build new dcd file
times=range(interval,steps+1,interval)
n=0
m=1
t=md.load_frame(dump_files[temp],0,top="../data/data2.psf")
for time in times[1:]: 
    while frames[n+1] < time:
        n+=1
    t2=md.load_frame(dump_files[replica[n]],m,top="../data/data2.psf")
    t=md.join((t,t2))
    m+=1


t.save("DUMP_FILE.dcd")
