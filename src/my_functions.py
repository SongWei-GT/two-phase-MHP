
import numpy as np
import scipy as sp
from scipy.linalg import expm



def my_element_wise_multuplication(A,B):
    
    if A.shape[0] == 0:
        return np.array([])
    else:
        return np.multiply(A,B)


def exp_kernel_internsity(s,all_event_time, all_event_idx, mu, A, B):
    """
    outputs
    ---------------
    this function calculates the intensity given complete history
    
    lambda_vec[i] is the conditional intensity of i-th process at time s


    inputs
    ---------------      
    all_event_time[i] is the exact event occurence time of i-th event
    all_event_idx[i] is the index of that event
    
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    """


    
    d = A.shape[0]
    lambda_vec = np.zeros(d)
    
    if type(B) == float:
        
        for triggered_event_idx in range(d):
            
            if len(all_event_idx) == 0:
                lambda_vec[triggered_event_idx] = mu[triggered_event_idx]
            else:
                tmp_lag = B * (s-all_event_time)
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[triggered_event_idx,all_event_idx] ,np.exp(- tmp_lag )))
                lambda_vec[triggered_event_idx] = mu[triggered_event_idx] + tmp_trigger_effect
        
        
    if (type(B) == np.matrix) |  (type(B) == np.ndarray):
    
        for triggered_event_idx in range(d):
            if len(all_event_idx) == 0:
                lambda_vec[triggered_event_idx] = mu[triggered_event_idx]
            else:      
                tmp_lag = my_element_wise_multuplication(B[triggered_event_idx,all_event_idx],s-all_event_time)
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[triggered_event_idx,all_event_idx] ,np.exp(- tmp_lag )))
                lambda_vec[triggered_event_idx] = mu[triggered_event_idx] + tmp_trigger_effect
    

    
    return lambda_vec




def multivariate_Hawkes_generate(T,mu,A,B, negative_intensity = False):
    """
    outputs
    ---------------
    this function simulates a MHP on time horizon T
    
    all_event_time[i] is the exact event occurence time of i-th event
    all_event_idx[i] is the index of that event
    
    
    
    inputs
    ---------------
    T is time horizon
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    """
    
    all_event_time = np.array([])
    all_event_idx = np.array([])
    all_event_idx = all_event_idx.astype('int')
    
    
    s=0 #start
    ite=1
    
    if negative_intensity == False:
        
        while s<=T:
            
            
        
            lambda_vec = exp_kernel_internsity(s,all_event_time, all_event_idx, mu, A, B)
            # we do not allow negative intensity        
            lambda_vec[lambda_vec < 0] = 0
            lambda_bar = sum(lambda_vec)
            u = np.random.uniform(0,1)
            w = -np.log(u)/lambda_bar
            s = s+w
            D = np.random.uniform(0,1)
            lambda_vec_new = exp_kernel_internsity(s,all_event_time, all_event_idx, mu, A, B)
            # we do not allow negative intensity        
            lambda_vec_new[lambda_vec_new < 0] = 0
            
            if D * lambda_bar <= sum(lambda_vec_new):
                
                all_event_time = np.append(all_event_time, s)
                
    
                new_event_idx = 0
                
                while D * lambda_bar > sum(lambda_vec_new[0:(new_event_idx+1)]):
                    
                    new_event_idx+=1
                    
                all_event_idx = np.append(all_event_idx, int(new_event_idx))
        
            
            ite += 1
        
    else:
        
        while s<=T:
            
            
        
            lambda_vec = exp_kernel_internsity(s,all_event_time, all_event_idx, mu, A, B)
            # we allow negative intensity        
            lambda_bar = sum(lambda_vec)
            u = np.random.uniform(0,1)
            w = -np.log(u)/lambda_bar
            s = s+w
            D = np.random.uniform(0,1)
            lambda_vec_new = exp_kernel_internsity(s,all_event_time, all_event_idx, mu, A, B)
            # we allow negative intensity        

            
            if D * lambda_bar <= sum(lambda_vec_new):
                
                all_event_time = np.append(all_event_time, s)
                
    
                new_event_idx = 0
                
                while D * lambda_bar > sum(lambda_vec_new[0:(new_event_idx+1)]):
                    
                    new_event_idx+=1
                    
                all_event_idx = np.append(all_event_idx, int(new_event_idx))
        
            
            ite += 1
            
    if all_event_time[-1]>T:
        return [all_event_time[0:(len(all_event_time)-1)] , all_event_idx[0:(len(all_event_time)-1)]]
    else:
        return [all_event_time , all_event_idx]



def generate_multi_seq_multivariate_Hawkes(T,mu,A,B,M, negative_intensity = False):
    """
    M is the number of sequences
    
    This functions generates M MHP sequences on time horizon T given parameters mu, A, B
    """
    all_event_time = []
    all_event_idx = []
    
    for i in range(M):
        
    
        [tmp_all_event_time , tmp_all_event_idx] = multivariate_Hawkes_generate(T,mu,A,B, negative_intensity = negative_intensity)
        
        all_event_time.append(tmp_all_event_time)
        all_event_idx.append(tmp_all_event_idx)
    
    return [all_event_time , all_event_idx]



def exp_kernel_log_lik(all_event_time, all_event_idx, mu, A, B):
    """
    outputs
    ---------------
    this function evaluates the log-likelihood for a single sequence
    
    inputs
    ---------------    
    all_event_time[i] is the exact event occurence time of i-th event
    all_event_idx[i] is the index of that event
    
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    """    
    
    
    d = A.shape[0]
    N = len(all_event_time)
    
    tmp_1 = 0
    
    tmp_2 = - max(all_event_time) * sum(mu)
    
    tmp_3 = 0
    

    if type(B) == float:

        
        for n in range(N):

            tmp_lag = B*(all_event_time[n]-all_event_time[0:n])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
            tmp_1 += np.log(mu[all_event_idx[n]] + tmp_trigger_effect)
            
            tmp_3 += sum(A[0:d,all_event_idx[n]] / B * (np.exp(-B*(all_event_time[N-1]-all_event_time[n]))-1))
            

        
    if (type(B) == np.matrix) |  (type(B) == np.ndarray):


        for n in range(N):

            tmp_lag = my_element_wise_multuplication(B[all_event_idx[n],all_event_idx[0:n]],all_event_time[n]-all_event_time[0:n])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
            tmp_1 += np.log(mu[all_event_idx[n]] + tmp_trigger_effect)
            
            
        
        for i in range(d):
            
            
            
            for n in range(N-1):
                tmp_lag_1 = my_element_wise_multuplication(B[i,all_event_idx[0:(n+1)]],all_event_time[n+1]-all_event_time[0:(n+1)])
                tmp_lag_2 = my_element_wise_multuplication(B[i,all_event_idx[0:(n+1)]],all_event_time[n]-all_event_time[0:(n+1)])
                
                tmp_factor = my_element_wise_multuplication(A[i,all_event_idx[0:(n+1)]],1/B[i,all_event_idx[0:(n+1)]])
            
            
            
                tmp_trigger_effect = sum(my_element_wise_multuplication(tmp_factor,np.exp(- tmp_lag_1 ) - np.exp(- tmp_lag_2 )))
            
                tmp_3 += tmp_trigger_effect
            

            
    return tmp_1+tmp_2+tmp_3



def exp_kernel_log_lik_for_multiple_seq(all_event_time, all_event_idx, mu, A, B, M):
    """
    outputs
    ---------------
    this function evaluates the log-likelihood for M sequences
    
    inputs
    ---------------    
    all_event_time[i] is occurence times for the i-th sequence
    all_event_idx[i] is indices for the i-th sequence
    
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    
    M is the number of sequences
    """        
    log_lik = 0
    
    for i in range(M):
        
        log_lik += exp_kernel_log_lik(all_event_time[i], all_event_idx[i], mu, A, B)
        
    return log_lik

        
def exp_kernel_log_lik_finite_memory(all_event_time, all_event_idx, mu, A, B, memory_length = 10):
    
    '''
    this function evaluates the log-likelihood for a single sequence considering finite memory depth
    '''
    
    d = A.shape[0]
    N = len(all_event_time)
    
    tmp_1 = 0
    
    tmp_2 = - max(all_event_time) * sum(mu)
    
    tmp_3 = 0

        
    for n in range(N):

        
        current_time = all_event_time[n]
            
        if current_time <= memory_length:
            
            tmp_lag = B*(current_time-all_event_time[0:n])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
            tmp_1 += np.log(mu[all_event_idx[n]] + tmp_trigger_effect)
            
            tmp_3 += sum(A[0:d,all_event_idx[n]] / B * (np.exp(-B*(all_event_time[N-1]-all_event_time[n]))-1))
                
        else: 
            
            complete_history = all_event_time[0:(n+1)]
            start_index = np.where(complete_history > current_time - memory_length)[0][0]
            tmp_lag = B*(current_time-all_event_time[start_index:n])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
            tmp_1 += np.log(mu[all_event_idx[n]] + tmp_trigger_effect)
            tmp_3 += sum(A[0:d,all_event_idx[n]] / B * (np.exp(-B*(all_event_time[N-1]-all_event_time[n]))-1))



            
    return tmp_1+tmp_2+tmp_3



def exp_kernel_log_lik_for_multiple_seq_finite_memory(all_event_time, all_event_idx, mu, A, B, M, memory_length = 10):
    
    '''
    this function evaluates the log-likelihood for M sequences considering finite memory depth
    '''    
    log_lik = 0
    
    for i in range(M):
        
        log_lik += exp_kernel_log_lik_finite_memory(all_event_time[i], all_event_idx[i], mu, A, B, memory_length = memory_length)
        
    return log_lik
            

#%% the functions below are only useful when one uses log-barrier penlty
# However, our numerical simulations showed that it is not useful for recovering the problem parameters


def find_min_intensity(all_event_time, all_event_idx, mu, A, B):
 
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    tmp_intensity_vec = np.zeros(N) 
    
        
    for n in range(N):

        tmp_lag = B*(all_event_time[n]-all_event_time[0:n])
        tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
        tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
        
    return min(tmp_intensity_vec)

def find_min_intensity_for_multiple_seq(all_event_time, all_event_idx, mu, A, B, M):
    
    global_min = np.inf
    
    for i in range(M):
        
        tmp_min = find_min_intensity(all_event_time[i], all_event_idx[i], mu, A, B)
        if global_min >= tmp_min:
            global_min = tmp_min
            
    return global_min
    
def find_barrier(all_event_time, all_event_idx, mu, A, B, epsilon, M):
    
    global_min = find_min_intensity_for_multiple_seq(all_event_time, all_event_idx, mu, A, B, M)
    return (global_min - epsilon)


def find_min_intensity_finite_memory(all_event_time, all_event_idx, mu, A, B, memory_length = 10):
 
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    tmp_intensity_vec = np.zeros(N) 
    
        
    for n in range(N):
        
        current_time = all_event_time[n]
            
        if current_time <= memory_length:
            tmp_lag = B*(all_event_time[n]-all_event_time[0:n])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
            tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
        else: 
            complete_history = all_event_time[0:(n+1)]
            start_index = np.where(complete_history > current_time - memory_length)[0][0]
            tmp_lag = B*(current_time-all_event_time[start_index:n])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
            tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

                    
        


        
    return min(tmp_intensity_vec)

def find_min_intensity_for_multiple_seq_finite_memory(all_event_time, all_event_idx, mu, A, B, M, memory_length = 10):
    
    global_min = np.inf
    
    for i in range(M):
        
        tmp_min = find_min_intensity_finite_memory(all_event_time[i], all_event_idx[i], mu, A, B, memory_length = memory_length)
        if global_min >= tmp_min:
            global_min = tmp_min
            
    return global_min
    
def find_barrier_finite_memory(all_event_time, all_event_idx, mu, A, B, epsilon, M, memory_length = 10):
    
    global_min = find_min_intensity_for_multiple_seq_finite_memory(all_event_time, all_event_idx, mu, A, B, M, memory_length = memory_length)
    return (global_min - epsilon)
    
 
def find_min_intensity_new(T ,all_event_time, all_event_idx, mu, A, B):
    
    tmp_all_event_time = all_event_time.copy()
    tmp_all_event_idx = all_event_idx.copy()
    
    tmp_all_event_idx = tmp_all_event_idx[tmp_all_event_time < T]
    tmp_all_event_time = tmp_all_event_time[tmp_all_event_time < T]
    
    d = A.shape[0]

    
    tmp_intensity_vec = np.zeros(d) 
    
    for d_idx in range(d):
        


        tmp_lag = B*(T-tmp_all_event_time)
        tmp_trigger_effect = sum(my_element_wise_multuplication(A[d_idx,tmp_all_event_idx],np.exp(- tmp_lag )))
        tmp_intensity_vec[d_idx] = (mu[d_idx] + tmp_trigger_effect)

    
    return min(tmp_intensity_vec)


def find_min_intensity_for_multiple_seq_new(T, all_event_time, all_event_idx, mu, A, B, M):
    
    global_min = np.inf
    
    for i in range(M):
        
        tmp_min = find_min_intensity_new(T, all_event_time[i], all_event_idx[i], mu, A, B)
        if global_min >= tmp_min:
            global_min = tmp_min
            
    return global_min
    
def find_barrier_new(all_event_time, all_event_idx, mu, A, B, epsilon, T, M):
    
    global_min = np.inf
    for t in np.linspace(0.1,T,10*T):
    
        tmp_global_min = find_min_intensity_for_multiple_seq_new(t, all_event_time, all_event_idx, mu, A, B, M)
        if tmp_global_min < global_min:
            global_min = tmp_global_min
    
    return (global_min - epsilon)
    

# %%

def exp_kernel_log_lik_derivative(all_event_time, all_event_idx, mu, A, B, log_barrier = False, barrier = None, lambda_lb = 0.1, memory_length = 10):
    '''
        outputs
    --------------- 
    This function returns the gradient w.r.t. mu and A for a single sequence
    pl_pmu is the gradient w.r.t. baseline/background intensity, which is a d-dimensional array
    pl_pA is the gradient w.r.t. excitation matrix A, which is a d-by-d array
    
    
    
    
        inputs
    ---------------    
    all_event_time[i] is the exact event occurence time of i-th event
    all_event_idx[i] is the index of that event
    
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    '''
    
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    
    pl_pmu = np.zeros(d)
    
    pl_pA = np.zeros([d,d])
    

    if (log_barrier == True) & (lambda_lb > 0):

        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in range(N):
            
            
            
            current_time = all_event_time[n]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
     
                try:
                    complete_history = all_event_time[0:(n+1)]
                    start_index = np.where(complete_history > current_time - memory_length)[0][0]
                    tmp_lag = B*(current_time-all_event_time[start_index:n])
                    tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                    tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                except:
                    print("n=",n)
                    print("all_event_time=",all_event_time)
                    print("current time = ",current_time)
                    
                    return [ pl_pmu , pl_pA ]
                    
                    

        
        if barrier == None:
            
            barrier = 1  
            
            
        tmp_barriered_intensity_vec = tmp_intensity_vec.copy()
        
        tmp_barriered_intensity_vec -= barrier
        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        
        
        for i in range(d):
            
            I1 = (all_event_idx == i)
            
            tmp_1 = 1/tmp_intensity_vec[I1]
            
            tmp_2 = lambda_lb/tmp_barriered_intensity_vec[I1]
            #print("lam lb = ",lambda_lb)
            
            pl_pmu[i] = sum(tmp_1) + sum(tmp_2) - t_N
            ###

        for n in range(N):
            
            current_time = all_event_time[n]
            pl_pA[:,all_event_idx[n]] += (np.exp(-B*(t_N-current_time)) - 1)/B
            
            
            if current_time <= memory_length:
                for j in range(n):
                    tmp_3 = np.exp(-B*(current_time-all_event_time[j]))
                    
                    ###
                    tmp_4_lik = tmp_3 * (1/tmp_intensity_vec[n] )
                    tmp_4_barrier = tmp_3 * (lambda_lb/tmp_barriered_intensity_vec[n])
                    
                    tmp_4 = tmp_4_lik+tmp_4_barrier
                    ###
                    
                    pl_pA[all_event_idx[n],all_event_idx[j]] += tmp_4
            else:
                
                
                complete_history = all_event_time[0:(n+1)]
                
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                for j in range(start_index,n):
                    tmp_3 = np.exp(-B*(current_time-all_event_time[j]))
                    
                    ###
                    tmp_4_lik = tmp_3 * (1/tmp_intensity_vec[n] )
                    tmp_4_barrier = tmp_3 * (lambda_lb/tmp_barriered_intensity_vec[n])
                    
                    tmp_4 = tmp_4_lik+tmp_4_barrier
                    ###
                    
                    pl_pA[all_event_idx[n],all_event_idx[j]] += tmp_4

    else:
        # we penalize zero intensity in the following way
        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in range(N):
    
            current_time = all_event_time[n]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
                try:
                    complete_history = all_event_time[0:(n+1)]
                    start_index = np.where(complete_history > current_time - memory_length)[0][0]
                    
                    tmp_lag = B*(current_time-all_event_time[start_index:n])
                    tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                    tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                except:
                    print("n=",n)
                    print("all_event_time=",all_event_time)
                    print("current time = ",current_time)
                    
                    return [ pl_pmu , pl_pA ]
                # complete_history = all_event_time[0:n]
                # start_index = np.where(complete_history > current_time - memory_length)[0][0]
                # tmp_lag = B*(current_time-all_event_time[start_index:n])
                # tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                # tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
        
        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        
        for i in range(d):
            
            I1 = (all_event_idx == i)
            
            tmp_1 = 1/tmp_intensity_vec[I1]
            
            pl_pmu[i] = sum(tmp_1) - t_N
            
        for n in range(N):
            current_time = all_event_time[n]
            pl_pA[:,all_event_idx[n]] += (np.exp(-B*(t_N-all_event_time[n])) - 1)/B
            
            if current_time <= memory_length:
                for j in range(n):
                    
                    pl_pA[all_event_idx[n],all_event_idx[j]] += np.exp(-B*(current_time-all_event_time[j]))/tmp_intensity_vec[n]    
            else:
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                for j in range(start_index,n):
                    
                    pl_pA[all_event_idx[n],all_event_idx[j]] += np.exp(-B*(current_time-all_event_time[j]))/tmp_intensity_vec[n]    

    
    return [ pl_pmu , pl_pA ]





def my_penalized_SGD_with_log_barrier_update(all_event_time, all_event_idx, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01 ,penalty = None, lambda_1 = 0.01, lambda_DAG = 0.01, memory_length = 10):
    '''
        outputs
    --------------- 
    This function returns the (average) gradient w.r.t. mu and A over a small batch of sequences (out of M sequences)
    pl_pmu is the gradient w.r.t. baseline/background intensity, which is a d-dimensional array
    pl_pA is the gradient w.r.t. excitation matrix A, which is a d-by-d array
    
    
    
    
        inputs
    ---------------    
    all_event_time[i] is the exact event occurence time of i-th event
    all_event_idx[i] is the index of that event
    
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    
    
    We allow multiple regularizations, choices include: penalty ={None,'l1','DAG','both'}
    '''

    
    tmp_idx = list(range(M))
     
    np.random.shuffle(tmp_idx)
    
    pl_pmu = []
    pl_pA = []

    if log_barrier:
        
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            [ tmp_pl_pmu , tmp_pl_pA ] = exp_kernel_log_lik_derivative(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
           
            pl_pmu.append(tmp_pl_pmu)
            pl_pA.append(tmp_pl_pA)
 

    else:
            
    
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            [ tmp_pl_pmu , tmp_pl_pA ] = exp_kernel_log_lik_derivative(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
            
            pl_pmu.append(tmp_pl_pmu)
            pl_pA.append(tmp_pl_pA)
        
    
    mu_grad = np.mean(pl_pmu,0)
    A_grad = np.mean(pl_pA,0)
    
    # normalize the gradient
    #mu_grad /= np.linalg.norm(mu_grad)
    #A_grad /= np.linalg.norm(A_grad)
    
    
    if penalty == 'l1':
     
        lasso_grad = A*0 
        lasso_grad[A>0] = 1
        lasso_grad[A<0] = -1
        np.fill_diagonal(lasso_grad, 0)
        #lasso_grad /= np.linalg.norm(lasso_grad)
        
        A_grad -= lambda_1 * lasso_grad

    
    elif penalty == 'DAG':

        tilde_A = A.copy()
        tilde_A[A < 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A<0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad -= lambda_DAG * tmp_h_grad
        
        
        tilde_A = -A.copy()
        tilde_A[A > 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A>0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad += lambda_DAG * tmp_h_grad  
        
    elif penalty == 'both':   
        
        lasso_grad = A*0 
        lasso_grad[A>0] = 1
        lasso_grad[A<0] = -1
        np.fill_diagonal(lasso_grad, 0)
        #lasso_grad /= np.linalg.norm(lasso_grad)
        
        A_grad -= lambda_1 * lasso_grad
        
        tilde_A = A.copy()
        tilde_A[A < 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A<0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad -= lambda_DAG * tmp_h_grad
        
        
        tilde_A = -A.copy()
        tilde_A[A > 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A>0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad += lambda_DAG * tmp_h_grad  
        
    
    return [mu_grad, A_grad]

def my_penalized_GD_with_log_barrier_update(all_event_time, all_event_idx, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01 ,penalty = None, lambda_1 = 0.01, lambda_DAG = 0.01, memory_length = 10):
    '''
        outputs
    --------------- 
    This function returns the (average) gradient w.r.t. mu and A over all M sequences
    This is the GD version of the above SGD update, in practice we simple call the function with batch_size = M
    pl_pmu is the gradient w.r.t. baseline/background intensity, which is a d-dimensional array
    pl_pA is the gradient w.r.t. excitation matrix A, which is a d-by-d array
    
    
    
    
        inputs
    ---------------    
    all_event_time[i] is the exact event occurence time of i-th event
    all_event_idx[i] is the index of that event
    
    mu is a row vec corresponding to the baseline/background intebnsity
    A is the self- and mutual- excitation/inhibitation matrix
    B is a scalar decaying rate
    
    
    We allow multiple regularizations, choices include: penalty ={None,'l1','DAG','both'}
    '''

    
    tmp_idx = list(range(M))
     
    #np.random.shuffle(tmp_idx)
    
    pl_pmu = []
    pl_pA = []

    if log_barrier:
        
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            [ tmp_pl_pmu , tmp_pl_pA ] = exp_kernel_log_lik_derivative(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
           
            pl_pmu.append(tmp_pl_pmu)
            pl_pA.append(tmp_pl_pA)
 

    else:
            
    
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            [ tmp_pl_pmu , tmp_pl_pA ] = exp_kernel_log_lik_derivative(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
            
            pl_pmu.append(tmp_pl_pmu)
            pl_pA.append(tmp_pl_pA)
        
    
    mu_grad = np.mean(pl_pmu,0)
    A_grad = np.mean(pl_pA,0)
    
    # normalize the gradient
    #mu_grad /= np.linalg.norm(mu_grad)
    #A_grad /= np.linalg.norm(A_grad)
    
    
    if penalty == 'l1':
     
        lasso_grad = A*0 
        lasso_grad[A>0] = 1
        lasso_grad[A<0] = -1
        np.fill_diagonal(lasso_grad, 0)
        #lasso_grad /= np.linalg.norm(lasso_grad)
        
        A_grad -= lambda_1 * lasso_grad

    
    elif penalty == 'DAG':
        

        tilde_A = A.copy()
        tilde_A[A < 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A<0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad -= lambda_DAG * tmp_h_grad
        
        
        tilde_A = -A.copy()
        tilde_A[A > 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A>0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad += lambda_DAG * tmp_h_grad  
        
    elif penalty == 'both':   
        
        lasso_grad = A*0 
        lasso_grad[A>0] = 1
        lasso_grad[A<0] = -1
        np.fill_diagonal(lasso_grad, 0)
        #lasso_grad /= np.linalg.norm(lasso_grad)
        
        A_grad -= lambda_1 * lasso_grad
        
        tilde_A = A.copy()
        tilde_A[A < 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A<0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad -= lambda_DAG * tmp_h_grad
        
        
        tilde_A = -A.copy()
        tilde_A[A > 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A>0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad += lambda_DAG * tmp_h_grad  
        
    
    return [mu_grad, A_grad]


def exp_kernel_log_lik_derivative_A(all_event_time, all_event_idx, target_index, mu, A, B, log_barrier = False, barrier = None, lambda_lb = 0.1, memory_length = 10):
    '''
    This is part of the above gradient update code, where we only deal with gradient for A (in phase 2 optimization) for a single sequence
    We only optimize for the target_index-th row of A
    '''
    
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    
    #pl_pmu = np.zeros(d)
    
    pl_pA = np.zeros([1,d])
    
    target_event_idx_collection = np.where(all_event_idx == target_index)[0]
    

    

    #if type(B) == float:
    if (log_barrier == True) & (lambda_lb > 0):

        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in target_event_idx_collection:
            
            
            
            current_time = all_event_time[n]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[0:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
 
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                tmp_lag = B*(current_time-all_event_time[start_index:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

                    

        if barrier == None:
            
            barrier = 1  
            
            
        tmp_barriered_intensity_vec = tmp_intensity_vec.copy()
        
        tmp_barriered_intensity_vec -= barrier
        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        


        for n in range(N):
            
            current_time = all_event_time[n]
            pl_pA[0,all_event_idx[n]] += (np.exp(-B*(t_N-current_time)) - 1)/B
            
           
        for n in  target_event_idx_collection:
            current_time = all_event_time[n]
        
            if current_time <= memory_length:
                for j in range(n):
                    tmp_3 = np.exp(-B*(current_time-all_event_time[j]))
                    
                    ###
                    tmp_4_lik = tmp_3 * (1/tmp_intensity_vec[n] )
                    tmp_4_barrier = tmp_3 * (lambda_lb/tmp_barriered_intensity_vec[n])
                    
                    tmp_4 = tmp_4_lik+tmp_4_barrier
                    ###
                    
                    pl_pA[0,all_event_idx[j]] += tmp_4
            else:
                
                
                complete_history = all_event_time[0:(n+1)]
                
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                for j in range(start_index,n):
                    tmp_3 = np.exp(-B*(current_time-all_event_time[j]))
                    
                    ###
                    tmp_4_lik = tmp_3 * (1/tmp_intensity_vec[n] )
                    tmp_4_barrier = tmp_3 * (lambda_lb/tmp_barriered_intensity_vec[n])
                    
                    tmp_4 = tmp_4_lik+tmp_4_barrier
                    ###
                    
                    pl_pA[0,all_event_idx[j]] += tmp_4

    else:
        # we penalize zero intensity in the following way
        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in target_event_idx_collection:
    
            current_time = all_event_time[n]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[0:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 

                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        

        for n in range(N):
            current_time = all_event_time[n]
            pl_pA[0,all_event_idx[n]] += (np.exp(-B*(t_N-all_event_time[n])) - 1)/B
            
        for n in target_event_idx_collection:  
            current_time = all_event_time[n]
            if current_time <= memory_length:
                for j in range(n):
                    
                    pl_pA[0,all_event_idx[j]] += np.exp(-B*(current_time-all_event_time[j]))/tmp_intensity_vec[n]    
            else:
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                for j in range(start_index,n):
                    
                    pl_pA[0,all_event_idx[j]] += np.exp(-B*(current_time-all_event_time[j]))/tmp_intensity_vec[n]    


    
    return pl_pA[0,:]


def my_penalized_GD_with_log_barrier_update_A(all_event_time, all_event_idx, target_index, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01 ,penalty = None, lambda_1 = 0.01, lambda_DAG = 0.01, memory_length = 10):
    '''
    This is part of the above gradient update code, where we only deal with gradient for A (in phase 2 optimization) for M sequencea
    We only optimize for the target_index-th row of A
    '''

    
    tmp_idx = list(range(M))
     
    #np.random.shuffle(tmp_idx)
    
    pl_pA = []

    if log_barrier:
        
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            tmp_pl_pA  = exp_kernel_log_lik_derivative_A(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], target_index, mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
           
            pl_pA.append(tmp_pl_pA)
 

    else:
            
    
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            tmp_pl_pA  = exp_kernel_log_lik_derivative_A(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], target_index, mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
            
            pl_pA.append(tmp_pl_pA)
        
    
    A_grad = np.mean(pl_pA,0)
    
    # normalize the gradient
    #mu_grad /= np.linalg.norm(mu_grad)
    #A_grad /= np.linalg.norm(A_grad)
    
    
    if penalty == 'l1':
     
        lasso_grad = A*0 
        lasso_grad[A>0] = 1
        lasso_grad[A<0] = -1
        np.fill_diagonal(lasso_grad, 0)
        #lasso_grad /= np.linalg.norm(lasso_grad)
        
        A_grad -= lambda_1 * lasso_grad[target_index,:]

    
    elif penalty == 'DAG':
        
        # TODO
        # what should be the DAG penalty?
        
        tilde_A = A.copy()
        tilde_A[A < 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A<0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad -= lambda_DAG * tmp_h_grad[target_index,:]
        
        
        tilde_A = -A.copy()
        tilde_A[A > 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A>0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad += lambda_DAG * tmp_h_grad[target_index,:]
        
    elif penalty == 'both':   
        
        lasso_grad = A*0 
        lasso_grad[A>0] = 1
        lasso_grad[A<0] = -1
        np.fill_diagonal(lasso_grad, 0)
        #lasso_grad /= np.linalg.norm(lasso_grad)
        
        A_grad -= lambda_1 * lasso_grad[target_index,:]
        
        tilde_A = A.copy()
        tilde_A[A < 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A<0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad -= lambda_DAG * tmp_h_grad[target_index,:]
        
        
        tilde_A = -A.copy()
        tilde_A[A > 0] = 0
        #np.fill_diagonal(tilde_A, 0)
        # calculate the DAG penalizier's grad
        h_grad = expm(tilde_A).T
        # do not update diagnoal
        #np.fill_diagonal(h_grad, 0)
        # do not update negative entries
        tmp_h_grad = h_grad.copy()
        tmp_h_grad[A>0] = 0
        #tmp_h_grad /= np.linalg.norm(tmp_h_grad)
        
        A_grad += lambda_DAG * tmp_h_grad[target_index,:]  
        
    
    return  A_grad

def exp_kernel_log_lik_derivative_mu(all_event_time, all_event_idx, target_index, mu, A, B, log_barrier = False, barrier = None, lambda_lb = 0.1, memory_length = 10):
    '''
    This is part of the above gradient update code, where we only deal with gradient for mu (in phase 2 optimization) for a single sequence
    We only optimize for the target_index-th element of mu
    '''
    
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    
    pl_pmu = 0
    
    #pl_pA = np.zeros([1,d])
    
    I1 = (all_event_idx == target_index)
    
    target_event_idx_collection = np.where(I1)[0]
    

    

    #if type(B) == float:
    if (log_barrier == True) & (lambda_lb > 0):

        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in target_event_idx_collection:
            
            
            
            current_time = all_event_time[n]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
 
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                tmp_lag = B*(current_time-all_event_time[start_index:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

                    
                    
        
        #TODO
        
        if barrier == None:
            
            barrier = 1  
            
            
        tmp_barriered_intensity_vec = tmp_intensity_vec.copy()
        
        tmp_barriered_intensity_vec -= barrier
        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        
        

            
        tmp_1 = 1/tmp_intensity_vec[I1]
        
        tmp_2 = lambda_lb/tmp_barriered_intensity_vec[I1]
        #print("lam lb = ",lambda_lb)
        
        pl_pmu = sum(tmp_1) + sum(tmp_2) - t_N
    


    else:
        # we penalize zero intensity in the following way
        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in target_event_idx_collection:
    
            current_time = all_event_time[n]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 

                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:n])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:n]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        

            
        tmp_1 = 1/tmp_intensity_vec[I1]
        
        pl_pmu = sum(tmp_1) - t_N
            

    
    return pl_pmu


def my_penalized_GD_with_log_barrier_update_mu(all_event_time, all_event_idx, target_index, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01, memory_length = 10):
    '''
    This is part of the above gradient update code, where we only deal with gradient for mu (in phase 2 optimization) for M sequences
    We only optimize for the target_index-th element of mu
    '''

    
    tmp_idx = list(range(M))
     
    #np.random.shuffle(tmp_idx)
    
    pl_pmu = []

    if log_barrier:
        
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            tmp_pl_pmu  = exp_kernel_log_lik_derivative_mu(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], target_index, mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
           
            pl_pmu.append(tmp_pl_pmu)
 

    else:
            
    
        for i in range(batch_size):
                
            tmp_idx_i = tmp_idx[i]
            #print(tmp_idx_i)
            
            tmp_pl_pmu  = exp_kernel_log_lik_derivative_mu(all_event_time[tmp_idx_i], all_event_idx[tmp_idx_i], target_index, mu, A, B, log_barrier = log_barrier, barrier = barrier, lambda_lb = lambda_lb, memory_length = memory_length)
            
            pl_pmu.append(tmp_pl_pmu)
        
    
    mu_grad = np.mean(pl_pmu,0)
    
  
    
    return  mu_grad