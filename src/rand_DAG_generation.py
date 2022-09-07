import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import colors
from src import my_functions
import scipy as sp
from scipy.linalg import expm


# define the DAG constraint
h = lambda A: expm(A).trace() - A.shape[0]

def h_no_diag(A):
    tmp_A = A.copy()
    np.fill_diagonal(tmp_A, 0)
    return h(tmp_A)


def my_rand_DAG(d):
    '''
    choose your own seed :)
    '''
    tmp_seed_1 = 6789

    
    
    input_dim = d
    
    while True:
        
        np.random.seed(tmp_seed_1)
        
        # rand initialization
        A1 = (np.random.rand(input_dim,input_dim))
        np.fill_diagonal(A1, 0)   
        
        
        
        print("initialization is: h(A) =",h(A1))   
        #print("A = ",A1)
        # true_mat = A1
        # fig, ax = plt.subplots()
        # im = ax.imshow(true_mat)
        # plt.show()
        
        iter_num = 5000
        lr = 0.5
        h_traj = []
        
        for iter_idx in range(iter_num):
            
            # projected GD algorithm
            tmp_grad = expm(A1).T
            tmp_grad /= np.linalg.norm(tmp_grad)
            A1 -= lr * tmp_grad
            A1[A1 < 0] = 0
            
            tmp_h = h(A1)
            h_traj.append(tmp_h)    
            if tmp_h == 0:
                print("Success after iter",iter_idx)
                #success_num += 1
                break
        
        if tmp_h != 0:
            tmp_seed_1 = int(tmp_seed_1 + 1)
            continue
       
        print("after GD: h(A) =",h(A1))   
        #print("A = ",A1)
        # true_mat = A1
        # fig, ax = plt.subplots()
        # im = ax.imshow(true_mat)
        # plt.show()
        
        # plt.plot(h_traj)
        # plt.xlabel('num. of iter.')
        # plt.ylabel('h func. value')
        # #plt.show()
        # #plt.savefig(file_name + "log_lik.png")
        # plt.show()        
        
        
        # round the number
        
        A = np.round(A1*5)/5
        print("after GD: h(A) =",h(A))   
        #print("A = ",A)
        min(A[A>0])
        print("rank is",np.linalg.matrix_rank(A))
        # true_mat = A
        # fig, ax = plt.subplots()
        # im = ax.imshow(true_mat)
        # plt.show()
        # 
        A /= 2
        
    
        mu = np.abs(np.random.rand(1,d))[0] * 0.3
        mu = np.round(mu*5)/5
        
        mu /= 2
    
        beta = 0.8
        B = beta
        
        # remove the dimension without any element
        # while True:
        #     flag_rm_d = 0
        #     empty_idx_collection = []
        #     for tmp_idx in range(d):
        #         if (mu[tmp_idx] + np.sum(A[tmp_idx,:])) == 0:
        #             flag_rm_d = 1
        #             empty_idx_collection.append(tmp_idx)
            
        #     if flag_rm_d:
        #         print("We remove variable num.",empty_idx_collection)   
                
        #         tmpA = np.delete(A, empty_idx_collection, 1)
        #         A = np.delete(tmpA, empty_idx_collection, 0)
                
        #         mu = np.delete(mu, empty_idx_collection)
                
        #         d -= len(empty_idx_collection)
        #     else:
        #         break
            
        flag_rm_d = 0
        empty_idx_collection = []
        
        for tmp_idx in range(d):
            if (mu[tmp_idx] + np.sum(A[tmp_idx,:])) == 0:
                flag_rm_d = 1
                empty_idx_collection.append(tmp_idx)
                
        # we skip those random initializations with a missing event type    
        if flag_rm_d:
            
            tmp_seed_1 = int(tmp_seed_1 + 1)
            continue
        
        else:
     
            # we only allow some negative entries 
            tmp_order = np.argsort(-np.sum(A,1)) 
    
            A[tmp_order[0],tmp_order[1]] = -0.5
            A[tmp_order[1],tmp_order[2]] = -0.3
            #A[tmp_order[0],tmp_order[3]] = -0.3
            
            # tmp_order_new = np.argsort(-np.sum(A,1)) 
            
            # additional_nagetive_counts = np.random.randint(3)
            
            # for tmp_my_idx in range(additional_nagetive_counts):
                
            #     node_idx1 = tmp_order_new[np.random.randint(3)]
                
                
            #     max_val = np.max(A[node_idx1,:])
            #     tmp_val = int((max_val/2 + np.random.rand()/5)*10)/10
                
            #     total_val = np.sum(A[node_idx1,:])
                
            #     node_idx2 = np.random.randint(d)
                
            #     for tmp_tmp_my_idx in range(d):
    
            #         if (total_val - A[node_idx1,node_idx2] - tmp_val) > 0:
                        
            #             A[node_idx1,node_idx2] = -tmp_val
            #             break
                    
            #         else:
                        
            #             node_idx2 += 1
            #             node_idx2 = node_idx2%d
                        
            #             continue
            
            break
        
    return [mu, A]