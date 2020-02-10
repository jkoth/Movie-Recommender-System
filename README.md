# Movie Recommender System

## Render HTML files using below links:
- https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/01_Basic_Analysis.html
- https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/02_Collaborative_Filtering_Methods.html
- https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/03_CF_Evaluation.html
- https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/04_User_Interface.html

## Overview
Built a Recommender System Python application for the capstone project in Programming for Machine Learning course at DePaul University, Chicago. Application learns from user profile and returns personalized top ten movie recommendations. Application is built primarily for new users who will be prompted to rate twenty randomly selected movies and based on their responses, recommendations will be calculated. In case user chooses not to input ratings, user's demographic details are used to recommend movies. Finally, in case user neither rates movies nor provides valid demographics details, a default recommendation list will be returned.

## Process
### Phase One
Explored all data sets provided by MovieLens to identify their shape, size, and quality. Additionally, performed basic analysis on key data sets to find out data distribution especially, by demographics details. Since the application was going to utilize demographics details in calculating recommendations, it was important to have well spread data. Data sets were pre-cleaned by MovieLens hence, cleaning data wasn’t required. However, restructuring data sets in matrix form was required in order to use machine learning techniques. Users, Movies, and Ratings data sets were restructured as matrix to represent Users and Movies as row and column indices, respectively, while Ratings were represented at their intersection.

Jupyter Notebook HTML \- [Basic Analysis](https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/01_Basic_Analysis.html)


### Phase Two
Elected to use Collaborative Filtering (CF) technique in the application for generating movie recommendations. To further identify which type of CF to utilize, coded both, user-based and item-based CF functions in myModule.py module. Also coded multiple similarity calculation functions to be used in CF functions. Identified user-based CF as most efficient technique to use in the application for recommendations based on evaluations performed using permutations of different CF technique and similarity methods.

Jupyter Notebook HTML \- [CF Methods](https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/02_Collaborative_Filtering_Methods.html)

Jupyter Notebook HTML \- [CF Evaluation](https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/03_CF_Evaluation.html)


### Phase Three
Coded Python module, myModule_UI.py containing methods to support user interface script. Methods provide functionalities to capture user’s demographic details and movie ratings to be used for recommendation functions.

Jupyter Notebook HTML \- [UI](https://htmlpreview.github.io/?https://github.com/jkoth/Movie-Recommender-System/blob/master/04_User_Interface.html)



