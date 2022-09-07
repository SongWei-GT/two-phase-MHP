
import numpy as np
import scipy as sp
from scipy.linalg import expm



def my_element_wise_multuplication(A,B):
    
    if A.shape[0] == 0:
        return np.array([])
    else:
        return np.multiply(A,B)


def exp_kernel_internsity(s,all_event_time, all_event_idx, mu, A, B):
    
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

def exp_kernel_internsity_finite_memory(s,all_event_time, all_event_idx, mu, A, B, memory_length = 10):
    
    d = A.shape[0]
    lambda_vec = np.zeros(d)

        
    meaningful_idx = (all_event_time > s - memory_length)
    tmp_all_event_time = all_event_time[meaningful_idx]
    tmp_all_event_idx = all_event_idx[meaningful_idx]
        
    for triggered_event_idx in range(d):

        
        if len(tmp_all_event_idx) == 0:
            lambda_vec[triggered_event_idx] = mu[triggered_event_idx]
       
        else:
            
            tmp_lag = B * (s-tmp_all_event_time)
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[triggered_event_idx,tmp_all_event_idx] ,np.exp(- tmp_lag )))
            lambda_vec[triggered_event_idx] = mu[triggered_event_idx] + tmp_trigger_effect

    
    return lambda_vec

# %%

def multivariate_Hawkes_generate(T,mu,A,B, negative_intensity = False):
    """
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
    """
    all_event_time = []
    all_event_idx = []
    
    for i in range(M):
        
    
        [tmp_all_event_time , tmp_all_event_idx] = multivariate_Hawkes_generate(T,mu,A,B, negative_intensity = negative_intensity)
        
        all_event_time.append(tmp_all_event_time)
        all_event_idx.append(tmp_all_event_idx)
    
    return [all_event_time , all_event_idx]

# %%

        
def exp_kernel_log_lik_finite_memory(all_event_time, all_event_idx, mu, A, B, memory_length = 10):
    
    d = A.shape[0]
    N = len(all_event_time)
    
    tmp_1 = 0
    
    tmp_2 = - max(all_event_time) * sum(mu)
    
    tmp_3 = 0
    

        
    for n in range(N):

        current_time = all_event_time[n]
        tmp_idx = np.where(all_event_time == current_time)   
        history_end_idx = tmp_idx[0][0]
            
        if current_time <= memory_length:
            

            
    
            tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
            
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
            
            tmp_lik = mu[all_event_idx[n]] + tmp_trigger_effect
            
            if tmp_lik == 0:
                return np.nan
            
            tmp_1 += np.log(tmp_lik)
            
            tmp_3 += sum(A[0:d,all_event_idx[n]] / B * (np.exp(-B*(all_event_time[N-1]-current_time))-1))
                
        else: 
            
            complete_history = all_event_time[0:(n+1)]
            start_index = np.where(complete_history > current_time - memory_length)[0][0]
            
            
            tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
            
            tmp_lik = mu[all_event_idx[n]] + tmp_trigger_effect
            if tmp_lik == 0:
                return np.nan
            
            tmp_1 += np.log(tmp_lik)
            tmp_3 += sum(A[0:d,all_event_idx[n]] / B * (np.exp(-B*(all_event_time[N-1]-current_time))-1))


            
    return tmp_1+tmp_2+tmp_3



def exp_kernel_log_lik_for_multiple_seq_finite_memory(all_event_time, all_event_idx, mu, A, B, M, memory_length = 10):
    
    log_lik = []
    
    for i in range(M):
        
        tmp_log_lik = exp_kernel_log_lik_finite_memory(all_event_time[i], all_event_idx[i], mu, A, B, memory_length = memory_length)
        
        if (~np.isnan(tmp_log_lik)) & (~np.isinf(tmp_log_lik)):
            log_lik.append(tmp_log_lik)
        
    return np.mean(log_lik)
            


# %%

def find_min_intensity_finite_memory(all_event_time, all_event_idx, mu, A, B, memory_length = 10):
 
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    tmp_intensity_vec = np.zeros(N) 
    
        
    for n in range(N):
        
        current_time = all_event_time[n]
        tmp_idx = np.where(all_event_time == current_time)   
        history_end_idx = tmp_idx[0][0]
            
        if current_time <= memory_length:
            
            tmp_lag = B*(all_event_time[n]-all_event_time[0:history_end_idx])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
            tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
        else: 
            
            complete_history = all_event_time[0:(n+1)]
            start_index = np.where(complete_history > current_time - memory_length)[0][0]
            
            tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
            tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
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
    
# %%

def exp_kernel_log_lik_derivative(all_event_time, all_event_idx, mu, A, B, log_barrier = False, barrier = None, lambda_lb = 0.1, memory_length = 10):
    '''
    This function returns the gradient w.r.t. mu and A for a single sequence
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
            tmp_idx = np.where(all_event_time == current_time)   
            history_end_idx = tmp_idx[0][0]
            
            
            if current_time <= memory_length:
                
                tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
     

                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)


        
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
                    
                    tmp_time_gap = current_time-all_event_time[j]
                    
                    if tmp_time_gap > 0:
                    
                        tmp_3 = np.exp(-B*tmp_time_gap)
                        
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
                    tmp_time_gap = current_time-all_event_time[j]
                    
                    if tmp_time_gap > 0:
                        
                        tmp_3 = np.exp(-B*tmp_time_gap)
                        
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
            tmp_idx = np.where(all_event_time == current_time)   
            history_end_idx = tmp_idx[0][0]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
            
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

    
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
                    tmp_time_gap = all_event_time[n]-all_event_time[j]
                    if tmp_time_gap > 0:
                        
                        pl_pA[all_event_idx[n],all_event_idx[j]] += np.exp(-B*tmp_time_gap)/tmp_intensity_vec[n]    
            else:
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                for j in range(start_index,n):
                    tmp_time_gap = all_event_time[n]-all_event_time[j]
                    if tmp_time_gap > 0:
                        pl_pA[all_event_idx[n],all_event_idx[j]] += np.exp(-B*tmp_time_gap)/tmp_intensity_vec[n]    

    
    return [ pl_pmu , pl_pA ]




def my_penalized_GD_with_log_barrier_update(all_event_time, all_event_idx, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01 ,penalty = None, lambda_1 = 0.01, lambda_DAG = 0.01, memory_length = 10):
    '''
    This function returns the (average) gradient w.r.t. mu and A for multiple sequences
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

# %%

def exp_kernel_log_lik_derivative_A(all_event_time, all_event_idx, target_index, mu, A, B, log_barrier = False, barrier = None, lambda_lb = 0.1, memory_length = 10):
    '''
    This function returns the gradient w.r.t. A for a single sequence
    '''
    
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    

    pl_pA = np.zeros([1,d])
    
    target_event_idx_collection = np.where(all_event_idx == target_index)[0]
    

    if (log_barrier == True) & (lambda_lb > 0):

        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in target_event_idx_collection:
            
            
            
            current_time = all_event_time[n]
            tmp_idx = np.where(all_event_time == current_time)   
            history_end_idx = tmp_idx[0][0]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
 
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)


        if barrier == None:
            
            barrier = 1  
            
            
        tmp_barriered_intensity_vec = tmp_intensity_vec.copy()
        
        tmp_barriered_intensity_vec -= barrier
        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        
        

        

        for n in range(N):
            
            current_time = all_event_time[n]
            pl_pA[0,all_event_idx[n]] += (np.exp(-B*(t_N-current_time)) - 1)/B
            
           
        for n in target_event_idx_collection:
            
            current_time = all_event_time[n]
        
            if current_time <= memory_length:
                for j in range(n):
                    
                    tmp_time_gap = all_event_time[n]-all_event_time[j]
                    if tmp_time_gap > 0:
                        
                        tmp_3 = np.exp(-B*tmp_time_gap)
                        
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
                    
                    tmp_time_gap = all_event_time[n]-all_event_time[j]
                    if tmp_time_gap > 0:
                        tmp_3 = np.exp(-B*tmp_time_gap)
                        
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
            tmp_idx = np.where(all_event_time == current_time)   
            history_end_idx = tmp_idx[0][0]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 

                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[target_index,all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        

            
        for n in range(N):
            #current_time = all_event_time[n]
            pl_pA[0,all_event_idx[n]] += (np.exp(-B*(t_N-all_event_time[n])) - 1)/B
            
        for n in target_event_idx_collection:  
            
            current_time = all_event_time[n]
            if current_time <= memory_length:
                for j in range(n):
                    tmp_time_gap = all_event_time[n]-all_event_time[j]
                    if tmp_time_gap > 0:                    
                        pl_pA[0,all_event_idx[j]] += np.exp(-B*tmp_time_gap)/tmp_intensity_vec[n]    
            else:
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                for j in range(start_index,n):
                    tmp_time_gap = all_event_time[n]-all_event_time[j]
                    if tmp_time_gap > 0:                    
                        pl_pA[0,all_event_idx[j]] += np.exp(-B*tmp_time_gap)/tmp_intensity_vec[n]    


    
    return pl_pA[0,:]


def my_penalized_GD_with_log_barrier_update_A(all_event_time, all_event_idx, target_index, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01 ,penalty = None, lambda_1 = 0.01, lambda_DAG = 0.01, memory_length = 10):
    '''
    This function returns the (average) gradient w.r.t. A for multiple sequences
    We allow multiple regularizations, choices include: penalty ={None,'l1','DAG','both'}
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
    This function returns the gradient w.r.t. mu for a single sequence
    '''
    
    d = A.shape[0]
    N = len(all_event_time)
    
    t_N = max(all_event_time)
    
    
    pl_pmu = 0
    
    #pl_pA = np.zeros([1,d])
    
    I1 = (all_event_idx == target_index)
    
    target_event_idx_collection = np.where(I1)[0]
    


    if (log_barrier == True) & (lambda_lb > 0):

        tmp_intensity_vec = np.zeros(N) 
    
        
        for n in target_event_idx_collection:
            
            
            
            current_time = all_event_time[n]
            tmp_idx = np.where(all_event_time == current_time)   
            history_end_idx = tmp_idx[0][0]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 
 
                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
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
            tmp_idx = np.where(all_event_time == current_time)   
            history_end_idx = tmp_idx[0][0]
            
            if current_time <= memory_length:
                tmp_lag = B*(current_time-all_event_time[0:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[0:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)
                
            else: 

                complete_history = all_event_time[0:(n+1)]
                start_index = np.where(complete_history > current_time - memory_length)[0][0]
                
                tmp_lag = B*(current_time-all_event_time[start_index:history_end_idx])
                tmp_trigger_effect = sum(my_element_wise_multuplication(A[all_event_idx[n],all_event_idx[start_index:history_end_idx]],np.exp(- tmp_lag )))
                tmp_intensity_vec[n] = (mu[all_event_idx[n]] + tmp_trigger_effect)

        
        tmp_intensity_vec[tmp_intensity_vec <= 0] = np.inf
        

            
        tmp_1 = 1/tmp_intensity_vec[I1]
        
        pl_pmu = sum(tmp_1) - t_N
            

    
    return pl_pmu


def my_penalized_GD_with_log_barrier_update_mu(all_event_time, all_event_idx, target_index, mu, A, B, M, batch_size = 1, log_barrier = False, barrier = None, lambda_lb = 0.01, memory_length = 10):
    '''
    This function returns the (average) gradient w.r.t. mu for multiple sequences
    We allow multiple regularizations, choices include: penalty ={None,'l1','DAG','both'}
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