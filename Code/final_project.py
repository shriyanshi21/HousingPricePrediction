import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, BayesianRidge, Ridge
from xgboost import XGBRegressor
from mlxtend.regressor import StackingRegressor



# =============================================================================

#Reading the Data
trainingPath = 'data/train.csv'
initialDF = pd.read_csv(trainingPath)

 
# Digging more into the target variable i.e., SalePrice
def Target(initialDF):
    target = initialDF.loc[:, "SalePrice"]
    #showing the distribution of data using seaborn
    print(target.describe())

    sns.distplot(target)

# ============================================================================
# Data cleaning - conversion, standardization   
    
def checkCorr(initialDF):
    
    #corrmat = trainDF.corr()
    #correlations = corrmat["SalePrice"].sort_values(ascending=False)
    #features = correlations.index[0:10]
    #print(features)
    
    #sns.pairplot(trainDF[features], size = 2.5)
    #plt.show()
    
    
    # BEGIN: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: We are visualzing the correlation matrix here using a heatmap to represent 
    # different coefficients by different colors
    corrmat = initialDF.corr()
    f, ax = plt.subplots(figsize=(11,9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(initialDF[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    
    
    '''
    According to the heatmap Garage Cars and the Garage Area are the most correlated attribute followed by 
    YearBuilt and GarageYrBlt, GrLivArea and TotRmsAbsGr and 1stFlrSF and TotalBsmtSF
    
    '''
    
    most_corr = pd.DataFrame(cols)
    most_corr.columns = ['Most Correlated Features']
    print(most_corr)
    
    
    #We will check the collinearity between the attributes and eleminate the attribute that is not helpful in predicting sales price 
    
    corr1 = initialDF.loc[:, "GarageArea"].corr(initialDF.loc[:, "GarageCars"])
    corr2 = initialDF.loc[:, "YearBuilt"].corr(initialDF.loc[:, "GarageYrBlt"])
    corr3 = initialDF.loc[:, "GrLivArea"].corr(initialDF.loc[:, "TotRmsAbvGrd"])
    corr4 = initialDF.loc[:, "1stFlrSF"].corr(initialDF.loc[:, "TotalBsmtSF"])
    
    
    print("Collinearity between Garage Area and Garage Cars")
    print(corr1)
    print("Collinearity between Year Built and Garage Year Built")
    print(corr2)
    print("Collinearity between Ground Living Area and Total Rooms Above Ground")
    print(corr3)
    print("Collinearity between 1st Floor SF and Total Basement SF")
    print(corr4)
    

def checkOutliers(initialDF):
    
    initialDF = featureEngineering(initialDF)
    
    # Plotting a scatterplot of GrivArea vs SalePrice
    # BEGIN: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    # EXPLANATION: Dataframes are concatenated and plotted using bivariate analysis
    SPvsGrLivArea = pd.concat([initialDF.loc[:, 'SalePrice'], initialDF.loc[:, 'GrLivArea']], axis=1)
    SPvsGrLivArea.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000))
    
    # END: from https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
    
    '''
    By looking at the scatterplot we can conclude that people will pay more for more living area but the two 
    points on bottom right look off and so do the two points in the top right that are clustered together. 
    We will be removing the outliers manually in the handleOutliers function. 
    '''
    
    # Finding the Ids of the outliers in GrLivArea
    print(initialDF.sort_values(by = 'GrLivArea', ascending = False)[:4])
    
    SPvsYearsOld = pd.concat([initialDF.loc[:, 'SalePrice'], initialDF.loc[:, 'YearsOld']], axis=1)
    SPvsYearsOld.plot.scatter(x='YearsOld', y='SalePrice', ylim=(0,800000))
    
    '''
    The four points in the extreme right seem way too expensive for an old house. It can be dropped depending 
    on the performance of the model. 
    
    '''
    # Finding the Ids of the outliers in YearsOld
    yearsOldOutlier = (initialDF.loc[:, "YearsOld"] > 110) & (initialDF.loc[:, "SalePrice"] > 300000)
    print(initialDF.loc[yearsOldOutlier])
    
    SPvsLotArea = pd.concat([initialDF.loc[:, 'SalePrice'], initialDF.loc[:, 'LotArea']], axis=1)
    SPvsLotArea.plot.scatter(x='LotArea', y='SalePrice', ylim=(0,800000))
    
    '''
    The four points on the extreme right seem way out of the trend in the graph. We will be removing the outliers in the 
    handle outliers function. 
    '''
    # Finding the Ids of the outliers in vsLotArea
    print(initialDF.sort_values(by = 'LotArea', ascending = False)[:4])
    
    SPvsTotalBsmtSF = pd.concat([initialDF.loc[:, 'SalePrice'], initialDF.loc[:, 'TotalBsmtSF']], axis=1)
    SPvsTotalBsmtSF.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000))
    
    TotalBsmtOutlier = (initialDF.loc[:, "TotalBsmtSF"] > 6000)
    print(initialDF.loc[TotalBsmtOutlier])
    
    # This outlier was calculated above as well 
    
    '''
    There are probably other outliers in the training dataset but removing outliers is not always safe. It can
    effect the model badly if there are outliers in the test dataset as well. Instead of dropping the outliers we 
    will try to make some of our models more robust to tacle them. 
    '''

    
 
def handleOutliers(initialDF):
    
    # Dropping the 4 outliers in GrLivArea vs SalePrice and 2 outliers in YearsOld vs SalePrice
    droppingOutliers = (initialDF.loc[:, "Id"] == 1299) | (initialDF.loc[:, "Id"] == 524) | (initialDF.loc[:, "Id"] == 1183) | (initialDF.loc[:, "Id"] == 692) | (initialDF.loc[:, "Id"] == 186) | (initialDF.loc[:, "Id"] == 584) | (initialDF.loc[:, "Id"] == 314) | (initialDF.loc[:, "Id"] == 336) | (initialDF.loc[:, "Id"] == 250) | (initialDF.loc[:, "Id"] == 707)
    initialDF = initialDF.drop(initialDF[droppingOutliers].index)

    return initialDF
    

def checkMissingValues(initialDF):
    
    # Check the missing values in the entire training dataset
    # Count the values and sort it in descending order 
    total = initialDF.isnull().sum().sort_values(ascending=False)
    percent = (initialDF.isnull().sum()/initialDF.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    sns.heatmap(initialDF.isnull(), yticklabels=False,cbar=False)
    print("Missing Data", missing_data.head(20))
    
    '''There are attributes in the dataset that contain NaN values. Here, we are identifying the attributes with a lot 
    of missing information 
    '''
    for var in initialDF.columns:
        if initialDF[var].isnull().mean()>0.80:
            print(var, initialDF[var].unique())
    
    '''
    We tried to identify the variables that contain a lot of missing information (NaN) so that we can tacle that later 
    when we are dropping the attributes 
    '''
    
def handlingMissing(X):
    # We handled the missing values using the imputer
    # If the entire attribute does not have any values i.e, if the entire column consists of 
    # missing values we replace the value in the column by 0
    missing_all = False
    for col in X.columns:
        if X.loc[:, col].isnull().all():
            X.loc[:, col] = 0
    
    # BEGIN: from https://www.kaggle.com/dansbecker/handling-missing-values
    # EXPLANATION: Imputers can fill in missing values with mean values by default. Filling
    # in missing values is usually better than dropping them entirely because
    # those that are not missing might indicate some valuable patterns.
    imputer = SimpleImputer(missing_values = np.nan , strategy = 'mean')
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    # END: from https://www.kaggle.com/dansbecker/handling-missing-values
    return X

def featureEngineering(originalDF):
    
    # Replacing yearBuilt to yearsOld so that we can plot the attribute vs SalePrice and check if there are any outliers
    yearBuilt = originalDF.loc[:, "YearBuilt"]
    # subtracting by 2006 since the data was calculated between 2006 and 2010
    originalDF.loc[:, "YearsOld"] = 2006 - originalDF.loc[:, "YearBuilt"]
    
    # BEGIN: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    # EXPLANATION: We are replacing the categorical values such as BsmtQual with a numeric value between
    # 0 and the number of class minus 1 by the use of Label Encoder. It is being used because some categorial
    # value might contain some information in the dataset.
    EncoderList = ["KitchenQual", "GarageQual", "BsmtQual", "BsmtCond", "ExterQual", "GarageCond",
                     "OverallCond", "FireplaceQu", "ExterCond", "HeatingQC", "BsmtFinType1", "BsmtFinType2"]
	# Label Encoder
    for column in EncoderList:
        lbl = LabelEncoder()
        lbl.fit(list(initialDF[column].values))
        initialDF[column] = lbl.transform(list(initialDF[column].values))
    # END: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
    
    return originalDF

def standardize(df):
    
    Cols = df.select_dtypes(exclude=['object']).columns
    
    stds = df.loc[:, Cols].std()
    mean = df.loc[:, Cols].mean()

    df.loc[:, Cols] = (df.loc[:, Cols] - mean) / stds

    return df


# ===============================================================================


# The data is experimented with different pre-processing methods
def testResults(initialDF):
    
    # Build and test Grandient Boosting Regressor
    def testGBR(df, n_estimators=100, max_depth=3):
        y = df.loc[:, "SalePrice"]
        X = df.drop(["SalePrice"], axis=1)
        gbr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
        
        X = handlingMissing(X)
        
        cvScores = model_selection.cross_val_score(gbr, X, y, cv=10,
                                                   scoring = 'neg_mean_squared_error')
        print("\tMean Squared Error:", -1 * cvScores.mean())
      
    # Build and test Random Forest Regressor
    def testRF(df):
        y = df.loc[:, "SalePrice"]
        X = df.drop(["SalePrice"], axis=1)
        rf = RandomForestRegressor()
        
        X = handlingMissing(X)
        
        cvScores = model_selection.cross_val_score(rf, X, y, cv=10,
                                                   scoring = 'neg_mean_squared_error')
        print("\tMean Squared Error:", -1 * cvScores.mean())
    

    #Perform feature Engineering
    initialDF = featureEngineering(initialDF)
    
    # Dropping the Outliers
    initialDF = handleOutliers(initialDF)
    
    # Standardize the Data
    initialDF.loc[:, initialDF.columns != "SalePrice"] = standardize(initialDF.drop(["SalePrice"], axis=1))
    
    """
    One Hot Encoding and Binary encoding 
    """
    # One Hot Encoding
    colNames = ['SalePrice', 'LotArea', 'Neighborhood','OverallQual','GrLivArea','FullBath','GarageCars','YearsOld']
    trainPredictors = initialDF.loc[:, colNames]
    selected_hot_DF = pd.get_dummies(trainPredictors)
    print(selected_hot_DF)
    # The attributues with more than 80% missing values are dropped 
    most_hotDF = pd.get_dummies(initialDF.drop(['PoolQC', 'MiscFeature', 'Alley','Fence'], axis=1))
    # The attributes with more than 90% missing values and high correlation (GarageCars and GarageArea) values are dropped
    drop_corrDF = pd.get_dummies(initialDF.drop(['PoolQC', 'MiscFeature', 'Alley', 'GarageArea', 'YearBuilt'], axis=1))
    
    '''
    Displaying the Gradient Boosting Regressor and Random Forest Regressor Results
    '''
    
    print("Graident Boosting Regressor with 100 estimators ")
    print("Performing GBR on Selected Columns")
    testGBR(selected_hot_DF)
    print("Performing GBR on all Attributes except for ones with many missing values")
    testGBR(most_hotDF)
    print("Performing GBR on all Attributes except for ones with many missing values or high correlation")
    testGBR(drop_corrDF)
    
    print()
    
    
    print("Graident Boosting Regressor with 200 estimators")
    testGBR(selected_hot_DF, n_estimators=200)
    print(" Performing GBR on all Attributes except for ones that have many missing values")
    testGBR(most_hotDF, n_estimators=200)
    print("Performing GBR on all Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testGBR(drop_corrDF, n_estimators=200)
    
    # We can see that the Mean Squared Error is redured by increasing the estimators
    print()
    
    print("Random Forest Regressor")
    print("Performing GBR on Selected Columns")
    testRF(selected_hot_DF)
    print("Performing GBR on all Attributes except for ones that have many missing values")
    testRF(most_hotDF)
    print("Performing GBR on all Attributes except for ones that have many missing values or Highly Correlated Attributes")
    testRF(drop_corrDF)
    
    
# The best pre processing is done by One Hot Encoder in the above trails
def preprocess(initialDF):
    
    # Performing feature engineering
    initialDF = featureEngineering(initialDF)
    # Dropping the outliers
    initialDF = handleOutliers(initialDF)
    # Standardizing the Data
    initialDF.loc[:, initialDF.columns != "SalePrice"] = standardize(initialDF.drop(["SalePrice"], axis=1))
    
    # The attributues with more than 80% missing values are dropped 
    most_hotDF = pd.get_dummies(initialDF.drop(['PoolQC', 'MiscFeature', 'Alley','Fence'], axis=1))
    
    y = most_hotDF.loc[:, "SalePrice"]
    X = most_hotDF.drop(["SalePrice"], axis=1)
    X = handlingMissing(X)
    y = np.log1p(y)
    
    
    return X, y


# Storing X and y as Global variables 
X, y = preprocess(initialDF)


# BEGIN: https://github.com/Shitao/Kaggle-House-Prices-Advanced-Regression-Techniques/blob/master/code/single_model/base_model.py
# EXPLANATION: We want the search to evaluate the model's performance using RMSE and we do that by defining a function
# that creates a scorer for GridSearchCV

def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5

RMSE = make_scorer(mean_squared_error_, greater_is_better=False)
# END: https://github.com/Shitao/Kaggle-House-Prices-Advanced-Regression-Techniques/blob/master/code/single_model/base_model.py

# Rough hyperparameter tuning is performed one at a time and is done by XSEDE allocation
# Tuning the gradient boosting regressor
def tuneGBR(df):
    X, y = preprocess(df)
    
    """
    param_test1 = {'min_samples_split':range(2,20)}
    gsearch1 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_leaf = 1,
                max_depth = 4,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 0
                ), 
            param_grid = param_test1, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch1.fit(X, y)
    print("Best param and its score", gsearch1.best_params_, -gsearch1.best_score_, sep='\n')
    
    param_test2 = {'min_samples_leaf':range(2,10)}
    gsearch2 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_split = 17,
                max_depth = 4,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 0
                ), 
            param_grid = param_test2, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch2.fit(X, y)
    print("Best param and its score", gsearch2.best_params_, -gsearch2.best_score_, sep='\n')
    """
    """
    param_test3 = {'max_depth':range(2,10)}
    gsearch3 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_leaf = 3,
                min_samples_split = 17,
                max_features = 'sqrt',
                subsample = 0.8,
                random_state = 0
                ), 
            param_grid = param_test3, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch3.fit(X, y)
    print("Best param and its score", gsearch3.best_params_, -gsearch3.best_score_, sep='\n')
    """
    
    param_test4 = {'subsample': [0.76, 0.78, 0.82, 0.84]}
    gsearch4 = model_selection.GridSearchCV(
            estimator = GradientBoostingRegressor(
                learning_rate = 0.01,
                n_estimators = 1000,
                min_samples_leaf = 3,
                min_samples_split = 17,
                max_depth = 5, 
                max_features = 'sqrt',
                random_state = 0
                ), 
            param_grid = param_test4, 
            scoring = RMSE, 
            iid = False,
            cv = 10
        )
    gsearch4.fit(X, y)
    print("Best param and its score", gsearch4.best_params_, -gsearch4.best_score_, sep='\n')

# Making a dictionary of all the models so that we can calculated the RMSE later
models = {
          
          "gbr": GradientBoostingRegressor(n_estimators=1000,
                                           learning_rate=0.01, 
                                           min_samples_split=17, 
                                           min_samples_leaf=7, 
                                           max_depth=5, 
                                           max_features='sqrt', 
                                           subsample=0.8, 
                                           random_state=0), 
          
          "xgb": XGBRegressor(n_estimators=1000,
                              learning_rate=0.01, 
                              min_child_weight=3,
                              max_depth=6, 
                              gamma=0, 
                              subsample=0.8, 
                              colsample_bytree=0.85, 
                              random_state=0), 
          "rf": RandomForestRegressor(n_estimators=1000, 
                                      bootstrap=True, 
                                      max_features='sqrt', 
                                      max_depth=6, 
                                      min_samples_split=3, 
                                      min_samples_leaf=1, 
                                      random_state=0), 
          "knn": KNeighborsRegressor(n_neighbors = 10), 
          "ada": AdaBoostRegressor(n_estimators=1000,
                                   learning_rate=0.01, 
                                   loss='square', 
                                   random_state=0),
          # BEGIN: https://www.kaggle.com/gmishrakec/multi-regression-techniques/code
          # EXPLANATION: We are using the RobustScaler to make the lasso regression more Robust
          # since it is one of the regression model that is very sensitive to outliers 
          "lasso": make_pipeline(RobustScaler(), Lasso(alpha =0.0005, 
                                                       random_state=0)),
          # END: https://www.kaggle.com/gmishrakec/multi-regression-techniques/code
          "bayridge": BayesianRidge(), 
          "ridge": Ridge()
         }

# This function takes a model name in string
def tryModel(modelName):
    try:
        alg = models[modelName]
        X, y = preprocess(initialDF)
        cvScores = model_selection.cross_val_score(alg, X, y, cv=10, scoring = RMSE)
        
        print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))
    except:
        print("This particular model does not exist in the dictionary of models")



# BEGIN: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# EXPLANATION: We are calculating the final score by taking weighted average 
# of the predictions from each model. The code has been modified to calculate the weighted
# average instead of arithmetic average to get a better score
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        if len(weights) != len(models):
            print("ERROR: The number of models and weights do not match.")
            return
        if sum(weights) != 1.0:
            print("The sum of weights is required to be 1")
            return
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        #return np.mean(predictions, axis=1)
        predictionsDF = pd.DataFrame(data=predictions)
        weightedPredictions = predictionsDF.apply(lambda prediction: prediction * self.weights, axis=1)
        return np.sum(weightedPredictions, axis=1)
    
# END: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard


# An array of model names in string and a meta model name in string is taken as the parameters of the function
def tryStackingModels(modelNames, metaModelName):
    X, y = preprocess(initialDF)
    regs = [models[modelName] for modelName in modelNames]
    meta = models[metaModelName]
    
    stack = StackingRegressor(regressors = regs, 
                              meta_regressor = meta)

    cvScores = model_selection.cross_val_score(stack, X, y, cv=10, scoring = RMSE)
    
    print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))


# An array of model names in string is taken as the parameter
def tryAveragingModels(modelNames, pWeights):
    try:
        X, y = preprocess(initialDF)
        algs = tuple(models[modelName] for modelName in modelNames)
        # This gives an equal weight on every model
        # averaged_models = AveragingModels(models = algs, weights = [1/len(algs) for i in range(len(algs))])
        averaged_models = AveragingModels(models = algs, weights = pWeights)
        cvScores = model_selection.cross_val_score(averaged_models, X, y, cv=10, scoring=RMSE)
        
        print("RMSE: {:.6f} ({:.4f})".format(-cvScores.mean(), cvScores.std()))
    except:
        print("The specified models do not exist in the models.")


'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(initialDF):
    
    print("File Name?")
    filename = input()
    
    # Reading the test dataset
    testDF = pd.read_csv("data/test.csv")
    testIDs = testDF.loc[:, "Id"]
    
    # Dropping the Outliers
    initialDF = handleOutliers(initialDF)
    
    y = initialDF.loc[:, "SalePrice"]
    y = y.values
    trainDF = initialDF.drop(["SalePrice"], axis=1)

    # Concatinate test and initial (=train)
    train_numRow = trainDF.shape[0]
    dataset = pd.concat(objs=[trainDF, testDF], axis=0)
    
    # created full dataset using One-Hot Encoding and Feature Engineering
    
    dataset = pd.get_dummies(dataset)
    dataset = featureEngineering(dataset)
    
    # The numbers of features in the testing dataset and initial dataset (trainDF) needs to match 
    # So we split the dataset 
    trainDF = dataset.iloc[:train_numRow, :].copy()
    testDF = dataset.iloc[train_numRow:, :].copy()
    
    trainDF = standardize(trainDF) #standardizing the training dataset
    
    """ # Weighted Average
    modelNames = ["gbr", "xgb", "lasso", "ridge"]
    algs = tuple(models[modelName] for modelName in modelNames)
    model = AveragingModels(models = algs, weights = [0.24, 0.07, 0.46, 0.09])
    """
    
    model = models["ridge"]
    
    trainDF = handlingMissing(trainDF)
    model.fit(trainDF, y)   #fitting the model with trainig data

    # Preprocessing the test dataset 
    testDF = standardize(testDF)
    testDF = handlingMissing(testDF)

    predictions = model.predict(testDF)
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

