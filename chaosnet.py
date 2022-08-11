import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


def normalize_data(data):
    data_norm = (data - np.min(data))/np.float(np.max(data) - np.min(data))
    return data_norm


def iterations(q, a, b, c, length, check):
    #The function return a time series and its index values 
    timeseries = (np.zeros((length,2)))
    timeseries[0,0] = q
    for i in range(1, length):
        timeseries[i,0] = skew_tent((timeseries[i-1,0]), a, b, c, check)
        timeseries[i,1] = np.int(i)
    return timeseries



def probability_calculation(X_train, timeseries, b, epsilon):
    # Code for calculating tt-ss method based feature extraction
    M = X_train.shape[0]
    N = X_train.shape[1]

    probability = np.zeros((M,N))
    for i in range(0,M):
        for j in range(0,N):
            A = (np.abs((X_train[i,j]) - timeseries[:,0]) < epsilon)
            #firingtime[i,j] = timeseries[A.tolist().index(True),1]
            freq = (timeseries[0:np.int(timeseries[A.tolist().index(True),1]),0] - b < 0)
            if len(freq) == 0:
                probability[i, j] = 0
            else: 
                probability[i,j] = freq.tolist().count(False)/np.float(len(freq))

    return probability    


def chaos_method(X_train, timeseries, b, epsilon, method):
    if  method == "TT":
        # Code for calculating firing time
        M = X_train.shape[0]
        N = X_train.shape[1]

        firingtime = np.zeros((M,N))
        for i in range(0,M):
            for j in range(0,N):
                A = (np.abs((X_train[i,j]) - timeseries[:,0]) < epsilon)
                firingtime[i,j] = timeseries[A.tolist().index(True),1]
        return firingtime
    
    elif method == "TT-SS":
            # Code for calculating tt-ss method based feature extraction
        M = X_train.shape[0]
        N = X_train.shape[1]

        probability = np.zeros((M,N))
        for i in range(0,M):
            for j in range(0,N):
                A = (np.abs((X_train[i,j]) - timeseries[:,0]) < epsilon)
                #firingtime[i,j] = timeseries[A.tolist().index(True),1]
                freq = (timeseries[0:np.int(timeseries[A.tolist().index(True),1]),0] - b < 0)
                if len(freq) == 0:
                    probability[i, j] = 0
                else: 
                    probability[i,j] = freq.tolist().count(False)/np.float(len(freq))

        return probability   

def cosine_similar_measure(test_firingtime, y_test, a, b, c, avg_class_dist):
    
    i = 0
    y_pred_val = []
    sim = []
    tot_sim = []
    for a_val in test_firingtime:
        sim = []
        for b_val in avg_class_dist:
            sim.append(cosine_similarity(a_val.reshape(1,len(a_val)),b_val.reshape(1, len(b_val))))
        tot_sim.append(sim)
        y_pred_val.append(np.argmax(tot_sim[i]))
        i = i+1
    from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
    accuracy = accuracy_score(y_test, y_pred_val)*100
    recall = recall_score(y_test, y_pred_val , average="macro")
    precision = precision_score(y_test, y_pred_val , average="macro")
    f1 = f1_score(y_test, y_pred_val, average="macro")
    print("accuracy")
    print("%.3f" %accuracy)
    print("precision")
    print("%.3f" %precision)
    print("recall")
    print("%.3f" %recall)
    print("f1score")
    print("%.3f" %f1)
    from sklearn.metrics import confusion_matrix as cm
    cm = cm(y_test,y_pred_val)  
    print("Confusion matrix\n", cm)
    return y_pred_val

def class_avg_distance(DistMat, y_train, lab):
 
  
  samples = y_train.shape[0]
  P = np.count_nonzero(y_train == lab)
  Q = DistMat.shape[1]
  class_dist = np.zeros((P,Q))
  k =0
  for i in range(0, samples):
      if (y_train[i] == lab):
          class_dist[k,:]=DistMat[i,:]
          k = k+1
      
  return np.sum(class_dist, axis = 0)/class_dist.shape[0]


def skew_tent(x,a,b,c, check):
# b is the parameters of the map.a and c are 0 and 1 respectively.
# GLS maps are piece wise linear
# Based on the value of check- the function will return any of the two diffrent maps.  If Check = "Sk-T", 
# then skew-tent map is returned else skew-binary map is returned.
    if check == "Sk-T":
        if x < b:
            xn = ((c - a)*(x-a))/(b - a)
        else:
            xn = ((-(c-a)*(x-b))/(c - b)) + (c - a)
        return xn
    if check == "Sk-B":
        if x < b:
            xn = x/b
        else:
            xn = (1 - x)/(1 - b)
        return xn


def chaos_second_layer(X_train, y_train, q, a, b, c, length, check, timeseries, epsilon, q2, coeff0, coeff1, coeff2 ):

    M = X_train.shape[0]
    N = X_train.shape[1]

    if np.mod(X_train.shape[1],2) == 0:
        
        print("even number of features in the dataset")
        arr_dim = X_train.shape[1]
        layer_2_mat = np.zeros((X_train.shape[0], int(arr_dim/2) ))

        for i in range(0, M):
            jnew = 0
            for j in range(0, arr_dim, 2 ):
                #print("Row = ", i+1, "Column = ", j+1)
                y = []
                for k in range(j, j+2):

                    A = (np.abs((X_train[i,k]) - timeseries[:,0]) < epsilon)
                    y.append(timeseries[0 : A.tolist().index(True) + 1, 0])
                length_y0 = len(y[0])
                length_y1 = len(y[1])
                #print("length of y[0]", length_y0)
                #print("length of y[1]", length_y1)
                if length_y0 > length_y1:
                    y[1] = np.pad(y[1],  ( 0, length_y0 - length_y1 ), 'constant')
                elif length_y1 > length_y0:
                    y[0] = np.pad(y[0],  ( 0, length_y1 - length_y0 ), 'constant')


                z = coeff0 * y[0] + coeff1 * y[1]
                initial_val = q2

                l2_val = z[0] + coeff2 * initial_val

                l2_timeseries = np.zeros((len(z),1))
                l2_timeseries[0,0] = l2_val

                for r in range(1, len(z)):

                    l2_timeseries[r, 0] = skew_tent(l2_val, a , b , c , check)
                    l2_val = z[r] + coeff2 * l2_timeseries[r, 0]

                freq = (l2_timeseries - b < 0)
                if len(freq) == 0:
                    layer_2_mat[i, jnew] = 0
                    jnew = jnew+1
                else: 
                    layer_2_mat[i,jnew] = freq.tolist().count([False])/np.float(len(freq))
                    jnew = jnew + 1
                    
        return layer_2_mat
        
    else:
        print("odd number of features in the datasets")
        arr_dim = X_train.shape[1] - 1
        layer_2_mat = np.zeros((X_train.shape[0], int(arr_dim/2 + 1)))

        for i in range(0, M):
            jnew = 0
            for j in range(0, arr_dim, 2 ):
                #print("Row = ", i+1, "Column = ", j+1)
                y = []
                for k in range(j, j+2):

                    A = (np.abs((X_train[i,k]) - timeseries[:,0]) < epsilon)
                    y.append(timeseries[0 : A.tolist().index(True) + 1, 0])
                length_y0 = len(y[0])
                length_y1 = len(y[1])
                #print("length of y[0]", length_y0)
                #print("length of y[1]", length_y1)
                if length_y0 > length_y1:
                    y[1] = np.pad(y[1],  ( 0, length_y0 - length_y1 ), 'constant')
                elif length_y1 > length_y0:
                    y[0] = np.pad(y[0],  ( 0, length_y1 - length_y0 ), 'constant')


                z = coeff0 * y[0] + coeff1 * y[1]
                initial_val = q2

                l2_val = z[0] + coeff2 * initial_val

                l2_timeseries = np.zeros((len(z),1))
                l2_timeseries[0,0] = l2_val

                for r in range(1, len(z)):

                    l2_timeseries[r, 0] = skew_tent(l2_val, a , b , c , check)
                    l2_val = z[r] + coeff2 * l2_timeseries[r, 0]

                freq = (l2_timeseries - b < 0)
                if len(freq) == 0:
                    layer_2_mat[i, jnew] = 0
                    jnew = jnew + 1
                else: 
                    layer_2_mat[i,jnew] = freq.tolist().count([False])/np.float(len(freq))
                    jnew = jnew + 1
        layer_2_mat[:,-1] = X_train[:,-1]
        
        return layer_2_mat
