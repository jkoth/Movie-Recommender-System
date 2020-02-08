from numpy import *
from numpy import linalg as la
import numpy as np

#Euclidean Sim
#Takes two arrays and returns their similarity score between 0 and 1
def ecludSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

#Pearson Correlation Coefficient
#Takes two arrays and returns their similarity score between 0 and 1
def pearsSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

#Cossine Sim
#Takes two arrays and returns their similarity score between 0 and 1
def cosSim(inA,inB):
    num = float(np.sum(inA.T * inB))
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)                     

#Item-based collaborative filtering using standard estimation method
#required parameters:
#         dataMat - user-item matrix containing all the users and items
#         user    - user index number
#         simMeas - similarity function to be used 
#         iteam   - item index number
#Optional parameter: testMat - user-item matrix containing all test records 
def standEst(dataMat, user, simMeas, item, testMat=None):
    #Run this for validation
    if testMat != None:                                             #When performing validation, provide testMat
        n = shape(dataMat)[1]
        simTotal = 0.0; ratSimTotal = 0.0
        for j in range(n):
            userRating = testMat[user,j]
            if userRating == 0: continue
            overLap = nonzero(logical_and(testMat[:,item]>0, \
                                      dataMat[:,j]>0))[0]           #[0] to extract row indices from nonzero output
            if len(overLap) == 0: similarity = 0
            else: 
                similarity = simMeas(testMat[overLap,item], \
                                   dataMat[overLap,j])
                if np.isnan(similarity): similarity = 0.0             #pearsSim returning some values with nan          
            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0: return 0
        else: 
            return ratSimTotal/simTotal
    #If not validation, use this
    else:
        n = shape(dataMat)[1]
        simTotal = 0.0; ratSimTotal = 0.0
        for j in range(n):
            userRating = dataMat[user,j]
            if userRating == 0 or j == item: continue
            overLap = nonzero(logical_and(dataMat[:,item]>0, \
                                      dataMat[:,j]>0))[0]           #[0] to extract row indices from nonzero output
            if len(overLap) == 0: similarity = 0
            else: 
                similarity = simMeas(dataMat[overLap,item], \
                                   dataMat[overLap,j])
            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0: return 0
        else: 
            return ratSimTotal/simTotal

#User-based collaborative filter
#required parameters:
#         dataMat - user-item matrix containing all the users and items
#         user    - user index number
#         simMeas - similarity function to be used 
#         iteam   - item index number
#Optional parameter: testMat - user-item matrix containing all test records 
def standEst_UB(dataMat, user, simMeas, item, testMat=None):
    if testMat == None:
        dataMat = dataMat.T             #Transpose to Item User matrix
        n = shape(dataMat)[1]
        simTotal = 0.0; ratSimTotal = 0.0
        for j in range(n):
            simUserRat = dataMat[item,j]
            if simUserRat == 0 or j == user: continue
            overLap = nonzero(logical_and(dataMat[:,user]>0, \
                                          dataMat[:,j]>0))[0]           #[0] to extract row indices from nonzero output
            if len(overLap) == 0: similarity = 0
            else: 
                similarity = simMeas(dataMat[overLap,user], \
                                     dataMat[overLap,j])
            simTotal += similarity
            ratSimTotal += similarity * simUserRat
        if simTotal == 0: return 0
        else: 
            return ratSimTotal/simTotal
    else:
        dataMat = dataMat.T                                             #Transpose to Item User matrix
        testMat = testMat.T
        n = shape(dataMat)[1]
        simTotal = 0.0; ratSimTotal = 0.0
        for j in range(n):
            simUserRat = dataMat[item,j]                                #Rating by other users
            if simUserRat == 0: continue
            overLap = nonzero(logical_and(testMat[:,user]>0, \
                                          dataMat[:,j]>0))[0]           #[0] to extract row indices from nonzero output
            if len(overLap) == 0: similarity = 0
            else: 
                similarity = simMeas(testMat[overLap,user], \
                                     dataMat[overLap,j])
                if np.isnan(similarity): similarity = 0.0               #pearsSim returning some values with nan         
            simTotal += similarity
            ratSimTotal += similarity * simUserRat
        if simTotal == 0: return 0
        else: 
            return ratSimTotal/simTotal

#Singular Value Decomposition with the 95% default retention
#Run only once to save on computation        
def svdCalc(dataMat, ret_pct=0.95):
    #split the matrix using SVD from np.linalg
    U,Sigma,VT = la.svd(dataMat)
    
    #calc optimal singular values to keep in the matrix
    SigSq = np.square(Sigma)
    svdRet = SigSq.sum() * ret_pct
    SigSqSum = 0.0
    svd_count = 0
    for i in SigSq:
        SigSqSum  += i
        svd_count += 1
        if SigSqSum > svdRet: break
    #transform dataMat
    SigRet = mat(eye(svd_count)*Sigma[:svd_count])              #arrange retaind svd into a diagonal matrix
    xformedItems = dataMat.T * U[:,:svd_count] * SigRet.I       #create transformed items
    return xformedItems

#SVD estimation function
#required parameters:
#         dataMat - user-item matrix containing all the users and items
#         user    - user index number
#         simMeas - similarity function to be used 
#         iteam   - item index number
#         xformedItems - reduced demensional SVD dataset
def svdEst(dataMat, user, simMeas, item, xformedItems):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    data=mat(dataMat)
    for j in range(n):
        userRating = data[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: 
        return ratSimTotal/simTotal

#Evaluation
#required parameters:
#         dataMat - user-item matrix containing all training users and items
#         testMat - user-item matrix containing all test users and items
#Optional parameter: 
#         estMethod - default is standard estimation method
#         simMeas   - default is pearson correlation
#Returns MAE for evaluation
def validation(dataMat, testMat, estMethod=standEst, simMeas=pearsSim):
    user_err = []
    for user in range(testMat.shape[0]):
        rated_items_by_user = np.nonzero(testMat[user,:])[1]
        error_u = 0.0
        count_u = len(rated_items_by_user)
        for item in rated_items_by_user:
            estimatedScore = estMethod(dataMat, user, simMeas, item, testMat)
            error_u = error_u + abs(estimatedScore - testMat[user, item])
        user_err.append([error_u, count_u])
    sum = np.sum(user_err, axis=0)
    errs = sum[0]
    cnts = sum[1]
    MAE = errs / cnts
    return MAE
     
#This function returns a list of k-Nearest Neighbors based on the given parameters
#required parameters:
#         dataMat - user-item matrix containing all the users and items
#         user    - user index number for whome you want recommendations
#Optional parameter: 
#         estMethod - default is standard estimation method
#         simMeas   - default is Cosine similarity
#         N         - number of nearest neighbors
#         xformedItems - reduced demensional SVD dataset (required when using SVD estimation method)
def recommend(dataMat, user, xformedItems=None, N=3, simMeas=cosSim, estMethod=standEst):
    dataMat = np.asmatrix(dataMat)                  #use input as matrix data structure
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #find unrated items. Nonzero with given criteria returns indices with TRUE
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    if estMethod.__name__ == 'svdEst':
        #xformedItems = svdCalc(dataMat)            #Can be done from within the function
        for item in unratedItems:
            estimatedScore = estMethod(dataMat, user, simMeas, item, xformedItems)
            itemScores.append([item, estimatedScore])
    else:    
        for item in unratedItems:
            estimatedScore = estMethod(dataMat, user, simMeas, item)
            itemScores.append([item, estimatedScore])
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
