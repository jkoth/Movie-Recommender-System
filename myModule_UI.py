#This Module contains Python functions to support Recommender System's UI interaction

import pandas as pd
import numpy as np

"""
Purpose:
   - Returns an empty Array of shape (1 Row, cols Columns)
   - Columns size matches item (movie) counts in item data set; extracted at runtime
   - Returned array is used to capture ratings input by user for each randomly selected item
Param:
   - @cols: number of items available for ratings; captured programatically at runtime
"""
def newUser(cols):
    userProfile = np.zeros((1,cols))
    return userProfile

#builds message to display while requesting rating input from user
#required parameters:
#         data - dataframe containing items
#         i    - index number
def message(data, i):
    movie = data[1][i]
    msg = 'Rate movie (1 - 5) - ' + movie +' : '
    return msg

#error check for non-numeric and out of bound input
#'x' to exit the input mode
#required parameters:
#         msg - message with movie name for rating prompt
def newRating(msg):
    bad_input = True
    while bad_input:
        rating = input(msg)
        if rating == 'x':                       #allow user to skip rating 
            bad_input = False
            return None
        else:
            try:
                rating = int(rating)            #make sure input is integer
                bad_input = False
                if rating > 0 and rating <= 5:  #make sure input is between 1 and 5 (inclusive)
                    bad_input = False
                    return rating
                else: 
                    bad_input = True            #if errors, repeat prompt
                    print('Input Error - Rate only between 1 - 5; to skip rating, type \'x\'')
            except:
                bad_input = True                #if errors, repeat prompt
                print('Input Error - Rate only between 1 - 5; to skip rating, type \'x\'')

#update user profile with rating for given item
#required parameters:
#         user - user-item matrix containing all users and items, including new user
#         indx - new user's index number
#         rating - rating value to be assigned
def userUpdate(user, indx, rating):
    user[0][indx] = rating

#used to identify demographic group needed for default recomendations
def newUserDemo():
    print('\nFollowing details are needed for your user profile\n')
    age = input('Please enter age: ')
    gender = input('Please enter gender (M/F): ')
    occupation = input('Please enter occupation: ')
    zipcode = input('Please enter zip code: ')
    return int(age), gender.upper(), occupation.lower(), zipcode

#used to identify demographic group needed for default recomendations
#required parameters:
#         a - age to be discretized
def ageBin(a):
    if a < 16: return 'young'
    elif a >= 16 and a < 35: return 'young-adult'
    elif a >= 35 and a < 55: return 'mid-age'
    elif a >= 55: return 'old'
    else: return None
    
#Build the subset of demographic DF of users matching the new user' age, gender, and occupation
#required parameters:
#         data - original ratings dataset in the stacked form
#         uid  - array of index numbers of users matching new user's age, gender, and occupation
#Sorts the data by ratings column and returns top N records
def demoRec(data, uid, N=10):
    iterCt = 0
    for i in uid:
        if iterCt > 0:
            tmp = data[data[0]==i]
            matchDemo = pd.concat([matchDemo,tmp],axis=0,ignore_index=True)
        else:
            matchDemo = data[data[0]==i]
        iterCt += 1
    #sort the resulting DF to pick top N items by rating
    matchDemo_Sorted = matchDemo.sort_values(by=2,ascending=False)
    matchDemo_Sorted.reset_index(inplace=True, drop=True)       #this is helpful for looping
    #select top N to return
    return matchDemo_Sorted[:N]

#This will be used for default recomendations
#required parameters:
#         data - original ratings dataset in the stacked form
#Sorts the data by ratings column and returns top N records
def defRec(data, N=10):
    default_sorted = data.sort_values(by=2, ascending=False)
    default_sorted.reset_index(inplace=True, drop=True)
    return default_sorted[:N]


