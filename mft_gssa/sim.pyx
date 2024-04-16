import cython
import numpy as np
cimport numpy as np
import random
import sys
from time import time
from timeit import default_timer as timer
from libc.math cimport exp
from libc.math cimport log
from libc.math cimport fabs 
from libc.math cimport lgamma
import gc
sys.stdout.write("Starting Simulation" + '\n')
sys.stdout.flush()
start = timer()
initial_sum = 0
cdef int N= 40#60 
cdef int lenpop = 780 #1711#N*(N-1)/2 - (N-1)
cdef float Mc = 741
cdef int ntime
cdef int ncheck 

cdef int nsim = 1
cdef float T = 1.0
#cdef float epsilon 
cdef float beta = 1/T 


cdef float c2 = 1.0
cdef float cx1 = 100.0
cdef float cy1 = cx1
cdef float kc = 0.1

cdef float epsilon = -0.5555555556
cdef float e_cost = 0.0*epsilon - 1.0*epsilon #breaking goes final: 0, initial: 1
cdef float rate_break_like = kc*exp(-e_cost)
cdef float rate_break_unlike = kc*1.0
print(rate_break_like/kc, rate_break_unlike/kc)

sys.stdout.write( "c2 = %f cx1 = %f cy1 = %f kc = %f, e = %f" %(c2,cx1,cy1,kc,epsilon))
sys.stdout.write('\n')
sys.stdout.flush()
#cdef float test_cont = 1.0
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
cdef double mysum1D( np.ndarray[np.float64_t, ndim=1] myarray, int length_arr ):
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

cdef list lookup = []

for k in range(lenpop):
    lookup.append(OD2Doff(k,N,1))


@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef conrate(float maxmu):
    cdef int lenratelook = int(maxmu)
    cdef np.ndarray[np.float64_t, ndim=2] ratelook
    ratelook = np.zeros((2,lenratelook))
    
    cdef int isum
    cdef float muval,fudge, dF0, dF1, Qc, Qn, muval_next
    cdef float q_n = 1.0#0.0 
    cdef float q_c = 0.0#1.0
    #cdef float epsilon = 0.0#-200.0/360.0
    for isum in range(int(maxmu)):
        muval = isum
        muval_next = muval+1
        #muval = isum+1
        #muval_next = muval -1
    
        Qc =  muval/Mc #(mu-(N-1))/Mc
        Qn = muval_next/Mc #(mu+dq-(N-1))/Mc
        
        fudge = (
                    lgamma(Mc+1) - lgamma(muval_next+1)-lgamma(Mc-muval_next+1)
                    - (lgamma(Mc+1) - lgamma(muval+1)-lgamma(Mc-muval+1))
                )
       
        #print("current", muval, lgamma(Mc+1) - lgamma(muval+1)-lgamma(Mc-muval+1) )
        #print("next", muval -1 , lgamma(Mc+1) - lgamma(muval_next+1)-lgamma(Mc-muval_next+1) )
        #dF0 = ((100*(Qn-0.5)**4 -25*(Qn-0.5)**2 )-
        #          (100*(Qc-0.5)**4 -25*(Qc-0.5)**2 )
        #          -fudge
        #             )
        #dF1 = ((100*(Qn-0.5)**4 -25*(Qn-0.5)**2 + epsilon*q_n)-
        #          (100*(Qc-0.5)**4 -25*(Qc-0.5)**2 + epsilon*q_c)
        #         -fudge
        #             )
        dF0 = (( 664.043*Qn**2 - 6312.1*Qn**3 + 15000.0*Qn**4 +150.0*Qn)
              -(664.043*Qc**2 - 6312.1*Qc**3 + 15000.0*Qc**4 +150.0*Qc)
             +fudge
                 )
        
        dF1 = (( 664.043*Qn**2 - 6312.1*Qn**3 + 15000.0*Qn**4 +150.0*Qn)
              -(664.043*Qc**2 - 6312.1*Qc**3 + 15000.0*Qc**4 +150.0*Qc)
             +fudge
                 )
        
        #print(muval, fudge, dF1)    
        #print(isum, isum+1,Qc,Qn, fudge, dF0, dF1)
        #print(isum, muval, muval_next, fudge, dF0, dF1,exp(-dF0), exp(-dF1))
        #print("idx:", isum, "muval:",  muval, "next:", muval_next, 
        #      "dfudge:", fudge, "dF0:", dF0, "dF1:", dF1, "rF1:", exp(-dF0), "rdF2:",exp(-dF1))# "dF12:", dF12
              #,"dF12", exp(-dF12))
        #print("idx:", isum, "muval:",  muval, "next:", muval_next, "ratio mark/unmark:", kc*exp(-dF0)/rate_break_like, 
        #    "same marks")
        #print("idx:", isum, "muval:",  muval, "next:", muval_next, "ratio mark/unmark:", kc*exp(-dF0)/rate_break_unlike
        #     ,"not same marks")
        ratelook[0,isum] = kc*exp(-dF0)#i+1
        ratelook[1,isum] = kc*exp(-dF1) #i+1
        
    #ratelook[0][x] = e_ij = 0
    #rate_ook[1][x] = eij != 0
    
    #print(ratelook)
    return ratelook

ratelook = np.array(conrate(Mc))
#print(np.mean(ratelook[0]))

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef propensity( np.ndarray[np.float64_t, ndim=1] current_contacts, 
                np.ndarray[np.float64_t, ndim=1] current_marks,
                np.ndarray[np.float64_t, ndim=2] rates_tab):# float mu):
    
    lenpropen = 3*lenpop
    cdef np.ndarray[np.float64_t, ndim=2] propen = np.zeros((lenpropen, 6))
    
    #cdef np.ndarray[np.float64_t, ndim=1] current_contacts = contacts
    #cdef np.ndarray[np.float64_t, ndim=1] current_marks = marks
    #cdef np.ndarray[np.float64_t, ndim=2] rates_tab = ratelook
    #cdef np.ndarray[np.int_t, ndim=2] lkupcoord = np.array(lookup)
    cdef int k
    cdef int i
    cdef int j
    cdef float i_mark
    cdef float j_mark
    cdef float q_c
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
    cdef int rtab_idx = np.int(mu-(N-1))#np.int(mu-(N-1) -1)
    #cdef float fudge
    
    for k in range(lenpop):
        q_c = current_contacts[k]
        i,j = lookup[k]
        #j = lkupcoord[k,1]
        i_mark = current_marks[i]
        j_mark = current_marks[j]
        #if i_mark == j_mark and i_mark == 1.0:
        #    epsilon = 0.0
        #else:
        #    epsilon = 0.0
        
        
        if fabs(i-j) == 1.0:
            bb = 0.0
            
        else:
            bb = 1.0
        #contacts break: energy cost
        if q_c ==1 and bb == 1.0:
            dq = -1.0
            q_n = q_c + dq
            if i_mark == j_mark and i_mark == 1.0:
                rate =  rate_break_like
            else:
                rate = rate_break_unlike
            #rate =  kc
         
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
            
            #if i_mark == j_mark and i_mark == 1.0:
                #rate = rates_tab[1,rtab_idx]
            #else:
            rate = rates_tab[0,rtab_idx]
    
            #10^-20 
            #rate =  kc*exp(-dF) 
            propen[k+2*s, 0] = k
            propen[k+2*s, 1] = i
            propen[k+2*s, 2] = j
            propen[k+2*s, 3] = dq
            propen[k+2*s, 4] = rate
            propen[k+2*s, 5] = 0
                   
        if i_mark == 1:
            #print("x->y at location", i, "with", j)
            d_imark = -1
            if bb==0.0:
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
            if bb==0.0:
                rate =  0.5*c2*(1-i_mark) + q_c*cx1*(1-i_mark)*j_mark
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
            if bb==0.0:
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
            if bb==0.0:
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
  
    
    propen = erase_remove(propen, lenpropen)
    return propen


                  
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
cdef gillespie_draw(contacts, marks):
    cdef float props_sum
    cdef float mu
    cdef float invmu
    cdef float r1
    cdef float time
    cdef int rxn_index
    cdef float rxn
    cdef int idx   
   
    cdef np.ndarray[np.float64_t, ndim=2] props = propensity(contacts,marks, ratelook)
    
    cdef int lenprops = len(props)
    cdef np.ndarray[np.float64_t, ndim=1] rxn_probs = np.zeros(lenprops)
    props_sum = mysum(props, lenprops, 4)# np.sum(props[:,4])
    # Compute time
    
    r1 = np.random.random()
    time = (1.0/props_sum)*log(1.0/r1)
    
    #Compute discrete probabilities of each reaction
    for idx in range(lenprops):
        rxn_probs[idx] = props[idx,4]/props_sum
    #rxn_probs = props[:,4] / props_sum
    start_sample = timer()
    rxn_index = sample_discrete(rxn_probs, lenprops)
    
    rxn_event = [props[rxn_index], time]
    
    return rxn_event

@cython.boundscheck(False)  # Deactivate bounds checking                                                                  
@cython.wraparound(False)   # Deactivate negative indexing.                                                               
@cython.cdivision(True)     # Deactivate division by 0 checking.
cdef gillespie_ssa( contacts_0,  marks_0 , time_points,dt_list):
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
    contacts_out[0] = population_contacts
    marks_out[0] = population_marks
    
    while i < ntime:
        #print("Starting:", i)
        while t < time_points[i_time]:
            
            # draw the event and time step
            
            rxn_event, dt = gillespie_draw(population_contacts, population_marks)
            
            
            
            k_contact = rxn_event[0]
            i_mark = rxn_event[1]
            j_mark = rxn_event[2]
            delta = rxn_event[3]
            rxn_type = rxn_event[5]
            #dt_list.append([dt,delta, rxn_type])
            #if rxn_type = 0, then update population contacts at index k_contact add delta
            #if rxn_type = 1, then update population of marks at index i_mark add delta
            #if rxn_type = 2, then update population of marks at index j_mark add delta
            
            #Update the population
            population_contacts_previous = population_contacts.copy()
            population_marks_previous = population_marks.copy()
            #print("old contacts", population_contacts)
            #print("old marks", population_marks)
            if rxn_type == 0:
                #update the contacts
                #print("updating contact at k", k_contact, "i",i_mark, "j", j_mark, delta)
                population_contacts[k_contact] = population_contacts[k_contact]+delta
            elif rxn_type == 1:
                
                #print("updating mark at",  i_mark,  delta)
                population_marks[i_mark] = population_marks[i_mark]+delta
            elif rxn_type == 2:
                #print("updating mark at", j_mark, delta)
                population_marks[j_mark] = population_marks[j_mark]+delta
            
            dt_list.append([dt,np.sum(population_marks), np.sum(population_contacts)])
            #print("updated contacts", population_contacts)
            #print("updated marks", population_marks)
            #population[rxn_index] = population[rxn_index]+rxn
                
            # Increment time
            #dt_list.append(dt)
            t = t+ dt

        # Update the index
        i = np.searchsorted(time_points , t, 'right')
        #print("finished", i)
        
        #Update the population
        #pop_out[i_time:min(i,ntime)] = population_previous
        contacts_out[i_time:min(i,ntime)] = population_contacts_previous
        marks_out[i_time:min(i,ntime)] = population_marks_previous
        #np.save("testcheck_c.npy", contacts_out)
        #np.save("testcheck_m.npy", marks_out)
        i_time = i                   
    return contacts_out, marks_out 

def fixsum():
    init_arr = np.zeros(N)
    desired_sum = initial_sum
    chg_idx = np.random.choice(N, desired_sum, replace=False)
    for idx in chg_idx:
        init_arr[idx] = 1
    return init_arr

ntime = 1000
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
    for k in range(lenpop):
        ii,jj = lookup[k]
        if fabs(ii-jj) == 1.0:
            contacts_0[k] = 1
        else:
              continue
    #contacts_file = np.load("contacts_init05.npz")
    #contacts_0 = contacts_0['data'][10**5-1]
    #marks_file = np.load("marks_init05.npz")
    #marks_0 = marks_0['data'][10**5-1]
    #marks_0 = fixsum()
    
    #marks_0 = np.ones(N)
    marks_0 = np.zeros(N)
    #contacts_0 = np.zeros(lenpop)
    ncheck = 0
    while ncheck < 200:
        #print(contacts_0)
        #print(marks_0)
        #contacts_0 = contacts_file['init'][ncheck]
        #marks_0= marks_file['init'][ncheck]
        
        init_marks = np.sum(marks_0)
        init_contacts = np.sum(contacts_0)
        sys.stdout.write( "init_marks = %f, init_contacts = %f" %(init_marks, init_contacts))
        sys.stdout.write('\n')
        sys.stdout.flush()  
        
        sys.stdout.write("check: %i" % ncheck)
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        dt_list = []
        con_out, mark_out = gillespie_ssa(contacts_0, marks_0, time_points, dt_list)
        #np.savez_compressed("con_%i_split.npz" %ncheck, data=con_out)
        #np.savez_compressed("mark_%i_split.npz" %ncheck, data=mark_out)
        np.savez_compressed("dtlist_%i.npz" %ncheck, data=dt_list)
        sys.stdout.write("Saved: *_split.npz")
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        contacts_0 = con_out[ntime-1]
        marks_0 = mark_out[ntime-1]
        del con_out
        del mark_out
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
