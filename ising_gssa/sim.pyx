import cython
import numpy as np
cimport numpy as np
import random
import sys
import gc
import copy
from time import time
from timeit import default_timer as timer
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs 
from libc.math cimport lgamma
sys.stdout.write("Starting Simulation" + '\n')
sys.stdout.flush()
start = timer()
cdef int N=40
cdef int lenpop = 780#N*(N-1)/2
cdef int Mc = 741#N*(N-1)/2 -(N-1)
cdef int ntime
cdef int ncheck 
cdef int nsim = 1
cdef float T = 1.0
#cdef float beta = 1/T 
cdef float c2 = 1.0
cdef float cx1 = 100.0
cdef float cy1 = cx1
cdef float kc = 0.1
cdef float epsilon = -2.5
cdef float lambdaJ = 0.01
cdef float scan = 1.0
cdef float rate_like = kc*exp(-epsilon*(-1.0))
cdef float rate_unlike = kc
#from cython.parallel import prange
sys.stdout.write( "c2 = %f cx1 = %f cy1 = cx1 kc = %f, e = %f" %(c2,cx1,kc,epsilon))
sys.stdout.write('\n')
sys.stdout.flush()
##UTILITY FUNCTIONS
#Sum along axis, 1D sum, erase_remove idiom, 0D2D converter gives i,j given k
#Function to compute Earr and update Earr based on Jcoupling, contact population and last rxn
@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking
cdef double mysum( np.ndarray[np.float64_t, ndim=2] myarray, int length_arr,int axis):
    cdef int idx 
    cdef double sum_val = 0.0
    for idx in range(length_arr):
        sum_val = sum_val + myarray[idx,axis]
    return sum_val
@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking
cdef double mysum1D(np.ndarray[np.float64_t, ndim=1] myarray, int length_arr ):
    cdef int idx 
    cdef double sum_val = 0.0
    for idx in range(length_arr):
        sum_val = sum_val + myarray[idx]
    return sum_val

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef erase_remove( np.ndarray[np.float64_t, ndim=2] myarray, int length_arr):
    cdef int idx 
    cdef int cursor = 0
    
    for idx in range(length_arr):
    
        if myarray[idx,4]!=0:
            myarray[cursor,0] = myarray[idx,0]
            myarray[cursor,1] = myarray[idx,1]
            myarray[cursor,2] = myarray[idx,2]
            myarray[cursor,3] = myarray[idx,3]
            myarray[cursor,4] = myarray[idx,4]
            myarray[cursor,5] = myarray[idx,5]
            cursor = cursor +1
        else:
            continue
    return  myarray[0:cursor]

cdef OD2Doff( int k,  int N,  int offset):
    cdef int left = 0
    cdef int right
    cdef int i = 0
    cdef int M = N-offset
    cdef int t
    cdef int x_coord
    cdef int y_coord
    cdef int corcal
    cdef list out
    for i in range(M):
        right = 0
        for t in range(0, i+1):
            right += M-t
            #print(right)
        if left <= k < right:
            x_coord = i
            #print(x_coord)
            if i == 0:
                y_coord = k+offset
                #print(y_coord)
            else:
                corcal = 0
                for t in range(0, i):
                    corcal += M-t
                
                y_coord = k-corcal + i+offset
            break
            left = right
    out = [x_coord, y_coord]
    return out

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef earr_comp(np.ndarray[np.float64_t, ndim=1] contacts, int len_mod_cont, 
               np.ndarray[np.float64_t, ndim=2] jvals,
               np.ndarray[np.int64_t, ndim=1] k2g, np.ndarray[np.int64_t, ndim=1] g2k
              ):
    #this function is run once to compute earr based on init cont and jcouplings
    cdef int g, gp, k, kp
    cdef np.ndarray[np.float64_t, ndim=1] earr = np.zeros(len_mod_cont)
    for g in range(len_mod_cont):
        for gp in range(len_mod_cont):
            if g != gp:
                k = g2k[g]
                kp = g2k[gp]
                earr[g] = earr[g] + (jvals[g,gp])*contacts[kp]#*contacts[k]       
    return earr

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef earr_update(np.ndarray[np.float64_t, ndim=1] earr, 
                np.ndarray[np.float64_t, ndim=1] contacts, int len_mod_cont, int last_idx, int delta,
                np.ndarray[np.float64_t, ndim=2] jvals,
                np.ndarray[np.int64_t, ndim=1] k2g, np.ndarray[np.int64_t, ndim=1] g2k):
            #This is run multiple times to update Earr
            cdef int g,k,gmod
            gmod = k2g[last_idx]
            for g in range(len_mod_cont):
                if g !=gmod:
                    k = g2k[g]
                    earr[g] += delta*jvals[g,gmod]#*contacts[k]                
            return earr

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef calcCPB(np.ndarray[np.float64_t, ndim=1] contacts, #population of contacts 
             np.ndarray[np.int64_t, ndim=2] idx_list,  
             int numbeads):
    cdef int k,i,j,gidx
    cdef np.ndarray[np.float64_t, ndim=1] contperbead = np.zeros(numbeads)
    for k in range(len(contacts)):
        gidx = idx_list[k,0]
        i = idx_list[k,1]
        j = idx_list[k,2]
        #don't count bb contacts
        if gidx !=-1 and contacts[k] == 1: 
            contperbead[i] = contperbead[i]+1
            contperbead[j] = contperbead[j]+1
    return contperbead

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef updateCPB(np.ndarray[np.float64_t, ndim=1] contperbead, int i, int j, int delta):
    contperbead[i] = contperbead[i]+delta
    contperbead[j] = contperbead[j]+delta
    return contperbead 

####################################
#generate lookup tables for indices#
####################################
cdef list lookup = []

for k in range(lenpop):
    lookup.append(OD2Doff(k,N,1))

#loop through idx list using k in lenpop, if element i,0 is -1 then backbone contact, otherwise

idx_list = np.load("idx_list40.npz")
idx_list = np.array(idx_list['data'], dtype=int)
#idx is value of gidx (the gth modifiable contact), val is kidx the kth contact
g2k= [] 
for index in range(len(idx_list)):
    if idx_list[index][0] != -1:
        g2k.append(index)
g2k = np.array(g2k, dtype = int)
#idx is value of kidx, val is gidx, if val is -1, cont is bb not modifiable
k2g = idx_list[:,0]

hvals = np.load("h_test.npz")['h']
jvals = np.load("j_test.npz")['J']*scan

#print(idx_list.ndim, jvals.ndim, hvals.ndim)

#lookup_time = []
@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef propensity(np.ndarray[np.float64_t, ndim=1] current_contacts,
                np.ndarray[np.float64_t, ndim=1] current_marks,
                np.ndarray[np.float64_t, ndim=1] current_earr,
                np.ndarray[np.float64_t, ndim=1] current_contperbead, 
np.ndarray[np.float64_t, ndim=2] jvals, np.ndarray[np.float64_t, ndim=1] hvals, 
np.ndarray[np.int64_t, ndim=2] idx_list):# float mu):
    lenpropen = 3*lenpop
    cdef np.ndarray[np.float64_t, ndim=2] propen = np.zeros((lenpropen, 6))
    cdef int k, kprime
    cdef int i
    cdef int j
    cdef int gidx, gidxprime
    cdef float i_mark, d_imark
    cdef float j_mark, d_jmark
    cdef float q_c, qprime
    cdef float q_n
    cdef float dq
    cdef float rate
    cdef float dF
    cdef int s = 0
    cdef float bb
    cdef int index
    cdef float Qn
    cdef float Qc
    cdef float mu = mysum1D(current_contacts, lenpop)#np.sum(current_contacts)
    cdef float Jcont, Hcont,exvol
    
    #for k in prange(lenpop, nogil=True, num_threads = 2, schedule = 'dynamic'):
    for k in range(lenpop):
        q_c = current_contacts[k]
        gidx = idx_list[k,0]
        i = idx_list[k,1]
        j = idx_list[k,2]
        #i,j = lookup[k]
        i_mark = current_marks[i]
        j_mark = current_marks[j]
        #print("Computing Propensity for")
        #print("kval, i,j, gidx, curr con ", k,i,j, gidx, q_c )
        
        if gidx == -1: #is backbone
            bb = 0.0
        elif gidx != -1:#not backbone
            bb = 1.0
            
        if q_c ==1 and bb == 1.0:
            dq = -1.0
            q_n = q_c + dq
            
            if i_mark == j_mark and i_mark == 1.0:
                rate = rate_like#*exp(-e_cost)#rate_break_like
            else:
                rate = rate_unlike#*exp(-e_cost)#rate_break_unlike
            #rate =  kc
            #print("Computing breaking rate")
            #print("hval,ecost",hvals[gidx], e_cost )
            propen[k+2*s, 0] = k
            propen[k+2*s, 1] = i
            propen[k+2*s, 2] = j
            propen[k+2*s,3]  = dq
            propen[k+2*s, 4] = rate
            propen[k+2*s, 5] = 0
        #contacts make: entropy cost
        elif q_c == 0 and bb == 1.0:
            dq = +1.0
            q_n = q_c + dq
            #print("Computing making rate") 
            #start_lookup = timer()
            Jcont = 0.5*2.0*current_earr[gidx]*dq 
            Hcont = hvals[gidx]*dq
            exvol = lambdaJ*(current_contperbead[i]+ current_contperbead[j]-2*q_c)*dq*2 #+ 2*dq
            #end_lookup = timer()
            #lookup_time.append(end_lookup-start_lookup)
            rate = kc*exp(-(Jcont+Hcont+exvol))
 
            propen[k+2*s, 0] = k
            propen[k+2*s, 1] = i
            propen[k+2*s, 2] = j
            propen[k+2*s, 3] = dq
            propen[k+2*s, 4] = rate
            propen[k+2*s, 5] = 0
                   
        if i_mark == 1:
            #print("x->y at location", i, "with", j)
            d_imark = -1
            if bb==0.0 and i==0:
                rate = c2*i_mark + q_c*cy1*i_mark*(1-j_mark)
            elif bb == 0.0 and i !=0:
                rate = 0.5*c2*i_mark + q_c*cy1*i_mark*(1-j_mark)   
            else:
                rate = q_c*cy1*i_mark*(1-j_mark) 
                
            propen[k+2*s+1, 0] = k
            propen[k+2*s+1, 1] = i
            propen[k+2*s+1, 2] = j
            propen[k+2*s+1, 3] = d_imark
            propen[k+2*s+1, 4] = rate
            propen[k+2*s+1, 5] = 1
        
        elif i_mark == 0:
            #print("y-> x at location",i, "with", j)
            d_imark = +1
            if bb==0.0 and i==0:
                rate =  c2*(1-i_mark) + q_c*cx1*(1-i_mark)*j_mark
            elif bb == 0.0 and i !=0:
                rate = 0.5*c2*(1-i_mark) + q_c*cx1*(1-i_mark)*j_mark
            else:
                 rate = q_c*cx1*(1-i_mark)*j_mark
            propen[k+2*s+1, 0] = k
            propen[k+2*s+1, 1] = i
            propen[k+2*s+1, 2] = j
            propen[k+2*s+1, 3] = d_imark
            propen[k+2*s+1, 4] = rate
            propen[k+2*s+1, 5] = 1

        if j_mark == 1:
            #print("x-> y at location ", j,"with", i)
            d_jmark = -1
            if bb==0.0 and j==39:#N-1
                rate = c2*j_mark + q_c*cy1*j_mark*(1-i_mark)
            elif bb==0.0 and j!=39:
                rate = 0.5*c2*j_mark + q_c*cy1*j_mark*(1-i_mark)
            else:
                rate = q_c*cy1*j_mark*(1-i_mark)
            propen[k+2*s+2,0] = k
            propen[k+2*s+2,1] = i
            propen[k+2*s+2,2] = j
            propen[k+2*s+2,3] = d_jmark
            propen[k+2*s+2,4] = rate
            propen[k+2*s+2,5] = 2
        
        elif j_mark == 0:
            #print("y->x at location",j, "with", i)
            d_jmark = +1
            if bb==0.0 and j==39:
                rate = c2*(1-j_mark) + q_c*cx1*(1-j_mark)*i_mark
            elif bb==0.0 and j!=39:
                rate = 0.5*c2*(1-j_mark) + q_c*cx1*(1-j_mark)*i_mark
            else:
                rate = q_c*cx1*(1-j_mark)*i_mark
            propen[k+2*s+2,0] = k
            propen[k+2*s+2,1] = i
            propen[k+2*s+2,2] = j
            propen[k+2*s+2,3] = d_jmark
            propen[k+2*s+2,4] = rate
            propen[k+2*s+2,5] = 2
        
        s=s+1
  
    
    #propen = erase_remove(propen, lenpropen)
    return erase_remove(propen, lenpropen)
                
#general function to sample indices of a list with given probabilities
@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef sample_discrete(np.ndarray[np.float64_t, ndim=1] probs, int length):
    # Generate random number
    #cdef int length = len(probs)
    cdef float q
    cdef int i
    cdef float p_sum
    
    q = np.random.rand()
    #print("random", q)
    # Find index 
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum = p_sum+ probs[i]
        i = i+1
        if i == length:
            return length -1 
            break       
    return i - 1

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
# General Function to draw time interval and choice of reaction
cdef gillespie_draw(contacts, marks, earr,contperbead):
    cdef float props_sum
    cdef float mu
    cdef float invmu
    cdef float r1
    cdef float time
    cdef int rxn_index
    cdef float rxn
    cdef int idx   
  
    cdef np.ndarray[np.float64_t, ndim=2] props = propensity(contacts,marks,earr,contperbead, jvals, hvals, idx_list) #ratelook)

    cdef int lenprops = len(props)
    cdef np.ndarray[np.float64_t, ndim=1] rxn_probs = np.zeros(lenprops)
    props_sum = mysum(props, lenprops, 4)# np.sum(props[:,4])
    # Compute time
    #print("propensity ok", props_sum)
    r1 = np.random.random()
    time = (1.0/props_sum)*log(1.0/r1)
    #print(r1,time, "random time ok")
    #Compute discrete probabilities of each reaction
    for idx in range(lenprops):
        rxn_probs[idx] = props[idx,4]/props_sum
    #rxn_probs = props[:,4] / props_sum
    start_sample = timer()
    rxn_index = sample_discrete(rxn_probs, lenprops)
    
    rxn_event = [props[rxn_index], time]
    
    return rxn_event
#dt_list = []
@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef gillespie_ssa(contacts_0,  marks_0, earr_0,contperbead_0,time_points,dt_list):
    #pop_out = np.empty((len(time_points), update.shape[1]))
    contacts_out = np.zeros((ntime, lenpop))
    marks_out    = np.zeros((ntime, N))
    # Initialize and perform simulation
    cdef int i_time = 1
    cdef int i = 0
    cdef float t = time_points[0]
    cdef float dt
    cdef int k_contact
    cdef int i_mark
    cdef int j_mark
    cdef int rxn_type
    population_contacts = contacts_0.copy()
    population_marks = marks_0.copy()
    earr = earr_0.copy()
    contperbead = contperbead_0.copy()
    contacts_out[0] = population_contacts
    marks_out[0] = population_marks
    
    while i < ntime:
        #print("Starting:", i)
        while t < time_points[i_time]:
            # draw the event and time step
            rxn_event, dt = gillespie_draw(population_contacts, population_marks, earr,contperbead)
            #dt_list.append(dt)
            k_contact = rxn_event[0]
            i_mark = rxn_event[1]#bead idx
            j_mark = rxn_event[2]
            delta = rxn_event[3]
            rxn_type = rxn_event[5]
            #if rxn_type = 0, then update population contacts at index k_contact add delta
            #if rxn_type = 1, then update population of marks at index i_mark add delta
            #if rxn_type = 2, then update population of marks at index j_mark add delta
            
            #Update the population
            population_contacts_previous = population_contacts.copy()
            population_marks_previous = population_marks.copy()
            #print("old contacts", population_contacts)
            #print("old Earr", earr)
            #print("old marks", population_marks)
            if rxn_type == 0:
                #update the contacts
                #print("updating contact at k", k_contact, "i",i_mark, "j", j_mark, delta)
                population_contacts[k_contact] = population_contacts[k_contact]+delta
                earr=earr_update(earr, population_contacts, Mc, k_contact, delta, jvals, k2g, g2k)
                contperbead = updateCPB(contperbead, i_mark,j_mark, delta)
            elif rxn_type == 1:
                #print("updating mark at",  i_mark,  delta)
                population_marks[i_mark] = population_marks[i_mark]+delta
            elif rxn_type == 2:
                #print("updating mark at", j_mark, delta)
                population_marks[j_mark] = population_marks[j_mark]+delta
            #print("updated contacts", population_contacts)
            #print("updated Earr", earr)
            #print("updated marks", population_marks)
            #population[rxn_index] = population[rxn_index]+rxn
            #Increment time
            #dt_list.append(dt)
            #dt_list.append([dt,np.sum(population_marks), np.sum(population_contacts), copy.deepcopy(population_contacts), copy.deepcopy(population_marks)])
            dt_list.append([dt,np.sum(population_marks), np.sum(population_contacts)])
            t = t+ dt

        # Update the index
        i = np.searchsorted(time_points , t, 'right')
        #print("finished", i)
        #Update the population
        #pop_out[i_time:min(i,ntime)] = population_previous
        contacts_out[i_time:min(i,ntime)] = population_contacts_previous
        marks_out[i_time:min(i,ntime)] = population_marks_previous
        i_time = i                   
    return contacts_out, marks_out 


ntime =10**3
time_points = np.arange(ntime, dtype=np.float)
def run_sim():
    # Initialize output array
    #pops_c = np.zeros((nsim, ntime,lenpop))
    #pops_m = np.zeros((nsim, ntime,N))
    #print(pops)
    #Run the calculations
    #for i in range(nsim):
    #init_condition    
    contacts_0=np.zeros(lenpop)
    #contacts_0 = np.ones(lenpop)
    #contacts_0 = np.random.randint(2, size=lenpop)
    #contacts_0 = np.array(contacts_0, dtype=np.float)
    for k in range(lenpop):
        ii,jj = lookup[k]
        if fabs(ii-jj) == 1.0:
            contacts_0[k] = 1
        else:
              continue
    marks_0 = np.ones(N)
    #marks_0 = np.zeros(N)
    #marks_0 = np.random.randint(2, size=N)
    #marks_0 = np.array(marks_0, dtype=np.float)
    
    ncheck = 0
    while ncheck < 3*10**2:
        #print(contacts_0)
        #print(marks_0)
        sys.stdout.write("check: %i" % ncheck)
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        earr_0 = earr_comp(contacts_0,Mc, jvals, k2g, g2k)
        contperbead_0 = calcCPB(contacts_0, idx_list, N)
        #print(contperbead_0)
        dt_list = []
        con_out, mark_out = gillespie_ssa(contacts_0, marks_0,earr_0, contperbead_0,time_points,dt_list)
        contacts_0 = con_out[ntime-1]
        marks_0 = mark_out[ntime-1]
        del con_out
        del mark_out
        #np.savez_compressed("testcon_%i_N40.npz" %ncheck, data=con_out)
        #np.savez_compressed("testmark_%i_N40.npz" %ncheck, data=mark_out)
        np.savez_compressed("dtlist_%i.npz" %ncheck, data=dt_list)
        sys.stdout.write("Saved: *_N40.npz")
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        #contacts_0 = con_out[ntime-1]
        #marks_0 = mark_out[ntime-1]
        #del con_out
        #del mark_out
        del dt_list
        gc.collect()
        ncheck = ncheck + 1
    return 
run_sim()
end = timer()
print("Time")
print(end-start)
print("Fin")
print("saved")
#print(np.mean(dt_list))
