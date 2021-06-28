import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")


    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainDF)
    
    print("training dataset size:",trainDF.shape)
    print("testing dataset size:",testDF.shape)
    
    print("The first five rows of the training dataset", trainDF.head())
    
    print("The data type of training dataset", trainDF.dtypes)
    
    print(checkCorr(trainDF))
    
    print(Target(trainDF))
    
    ZscoreOutlier(trainDF['LotArea'])
    
   
    '''
    checkMissingValues(trainDF)
    var = trainDF.loc["LotFrontage"]
    handlingMissing(var)
    #trainDF.loc[:, trainDF.columns != "SalePrice"] = standardize(trainDF.drop(["SalePrice"], axis=1))

    '''
def GBRTestTry(df):
  
    
    # BEGIN: from https://www.kaggle.com/dansbecker/learning-to-use-xgboost
    # EXPLANATION: This a check for inspecting missing values in the SalesPrice attribute 
    # The entire row is dropped is the value is missing. Inplace is set as true for the data frame to be mutated
    df.dropna(axis = 0, subset = ['SalePrice'], inplace = True)
    # END: from https://www.kaggle.com/dansbecker/learning-to-use-xgboost
    y = df.loc[:, "SalePrice"]
    X = df.loc[:, ["LotArea", "YearBuilt", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GarageArea", "YrSold", ]]
    
    gbr = GradientBoostingRegressor()
    # BEGIN: from https://www.kaggle.com/dansbecker/handling-missing-values
    # EXPLANATION: Imputers can fill in missing values with mean values by default. Filling
    # in missing values is usually better than dropping them entirely because
    # those that are not missing might indicate some valuable patterns.
    imputer = SimpleImputer(missing_values = np.nan , strategy = 'median')
    imputer = imputer.fit(X)
    X = imputer.transform(X)
    # END: from https://www.kaggle.com/dansbecker/handling-missing-values
    cvScores = model_selection.cross_val_score(gbr, X, y, cv=10, scoring = 'neg_mean_squared_error')
    
    print("Mean Squared Error:", -1 * cvScores.mean()) #multiplying with -1 to get rid of the negative value 

def ZscoreOutlier(initialDF):
    out=[]
    mean = np.mean(initialDF)
    sd = np.std(initialDF)
    for i in initialDF: 
        z = (i-mean)/sd
        if np.abs(z) > 4: 
            out.append(i)
    print("Outliers:",out)
    
 
    
def Target(initialDF):
    target = initialDF.loc[:, "SalePrice"]
    #showing the distribution of data using seaborn
    print(target.describe())

    sns.distplot(target)
    
    
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
    
    
   
    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw09 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)
    print(trainInput.describe())
    print("Training input dataset size:",trainInput.shape)
    


# ============================================================================
# Data cleaning - conversion, normalization

def checkMissingValues(originalDF):
    
    # Check the missing values in the entire training dataset
    # Count the values and sort it in descending order 
    total = originalDF.isnull().sum().sort_values(ascending=False)
    percent = (originalDF.isnull().sum()/originalDF.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print("Missing Data", missing_data.head(20))
    
    
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



def standardize(df):
    
    Cols = df.select_dtypes(exclude=['object']).columns
    
    stds = df.loc[:, Cols].std()
    mean = df.loc[:, Cols].mean()

    df.loc[:, Cols] = (df.loc[:, Cols] - mean) / stds

    return df

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    predictors = ['1stFlrSF', '2ndFlrSF', 'LotArea', 'YearBuilt', 'TotalBsmtSF', 'GarageArea', 'OverallQual', 'FullBath', 'GarageCars', 'OverallCond', 'BsmtFinSF1']
    # The attributes taken as predictors above are taken according the heatmap described in the word doc 
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors


# ===============================================================================


def testResults(initialDF):
    
    # Standardize the Data
    initialDF.loc[:, initialDF.columns != "SalePrice"] = standardize(initialDF.drop(["SalePrice"], axis=1))
    """
    One Hot Encoding and Binary encoding 
    """
    # One Hot Encoding
    colNames = ['SalePrice', 'LotArea', 'Neighborhood','OverallQual','GrLivArea','FullBath','GarageCars','YardSize','YearsOld']
    trainPredictors = initialDF.loc[:, colNames]
    selected_hot_DF = pd.get_dummies(trainPredictors)
    print(selected_hot_DF)
    # The attributues with more than 90% missing values are dropped 
    most_hotDF = pd.get_dummies(initialDF.drop(['PoolQC', 'MiscFeature', 'Alley'], axis=1))
    # The attributes with more than 90% missing values and high correlation are dropped
    drop_corrDF = pd.get_dummies(initialDF.drop(['PoolQC', 'MiscFeature', 'Alley', 'GarageArea', 'YearBuilt'], axis=1))
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(initialDF):
    
    testDF = pd.read_csv("data/test.csv")
    testIDs = testDF.loc[:, "Id"]
    
    y = initialDF.loc[:, "SalePrice"]
    trainDF = initialDF.drop(["SalePrice"], axis=1)

    # Concatinate test and initial (=train)
    train_numRow = trainDF.shape[0]
    dataset = pd.concat(objs=[trainDF, testDF], axis=0)
    # created full dataset using One-Hot Encoding 
    dataset = pd.get_dummies(dataset)
    # The numbers of features in the testing dataset and initial dataset (trainDF) needs to match 
    # So we split the dataset 
    trainDF = dataset.iloc[:train_numRow, :].copy()
    testDF = dataset.iloc[train_numRow:, :].copy()
    
    trainDF = standardize(trainDF) #standardizing the training dataset
    
    gbr = GradientBoostingRegressor() #calling the gradient boosting Regressor 
    
    trainDF = handlingMissing(trainDF)
    gbr.fit(trainDF, y)   #fitting the model with trainig data

    # Preprocessing the test dataset 
    testDF = standardize(testDF)
    testDF = handlingMissing(testDF)

    predictions = gbr.predict(testDF)
    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

if __name__ == "__main__":
    main()

