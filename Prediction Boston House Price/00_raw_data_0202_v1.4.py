
'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-

'''
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
from scipy import stats
import warnings
warnings.simplefilter('ignore')

plt.style.use('ggplot')
sns.set_style('white')
pd.set_option('display.max_columns', 10)


# Read in train and test data
# 我會習慣於把ID保留，某些情況下需要對ID進行進一步的處理(因爲ID是唯一PK,需可溯源)，另一個原因是index會涉及到排序，ID排序會造成大混亂
# df_train ⇒ train 後面會用到 df_all 裏面分割出df_
def importing_data():
    train = pd.read_csv('E:/DS_DA_prep/Kaggle/Predict house price/train.csv')
    test = pd.read_csv('E:/DS_DA_prep/Kaggle/Predict house price/test.csv')

    train_id = train['Id']
    test_id = test['Id']

    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)

    print('The shape of train data(without Id) is : {}'.format(train.shape))
    print('The shape of test data(without Id) is : {}'.format(test.shape))

    train_y = train['SalePrice']
    train_x = train.drop('SalePrice', axis=1)
    df_all = pd.concat([train_x, test], axis=0)
    print('The shape of all data is : {}'.format(df_all.shape))

    return train, test, train_id, test_id, train_y, train_x, df_all


# Checking for Missing Values
# na_stats.total ⇒ na_stats['total'] 會更便於閲讀（因爲IDE中會有顔色變化）
def check_missing(df):
    total_count = df.isnull().count()
    total_na_count = df.isnull().sum()
    total_na_per = df.isnull().sum() / len(df) * 100
    na_stats = pd.concat([total_count, total_na_count, total_na_per], axis=1,
                         keys=['total_count', 'total_na_count', 'total_na_per'])
    na_stats = na_stats[na_stats['total_na_count'] > 0].sort_values(by='total_na_per', ascending=False)

    return na_stats


# Check for distribution (for y or others)
# 添加了歪度和尖度
def check_distribution(df, check_list):
    num_feature_list = df.dtypes[df.dtypes != 'object'].index
    skew_score = df[num_feature_list].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print('---------------------------')
    print('skew score of numerical features:')
    print('---------------------------')
    skewness = pd.DataFrame({'skew_score': skew_score})
    print(skewness)

    for feature in check_list:
        try:
            sns.distplot(df[feature], fit=norm)
            fig = plt.figure()
            res = stats.probplot(df[feature], plot=plt)

            print('---------------------------')
            print('Skewness of " %s " is: %f' % (feature, df[feature].skew()))
            print('Kurtosis of " %s " is: %f' % (feature, df[feature].kurt()))
            print('---------------------------')
        except ValueError:
            print(str(feature) + 'ValueError')


# Check for numeric variable correlations
def check_correlations(df, check_type='num_feature', target='SalePrice'):
    if check_type == 'num_feature':
        check_list = df.dtypes[df.dtypes != 'object'].index
        corrmat = df[check_list].corr()
        plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=0.9, square=True)
        plt.show()

        sns.set()
        sns.pairplot(df[check_list], size=2.5)
        plt.show()

    elif check_type == 'target':
        k = 10  # number of features for heatmap
        corrmat = df.corr()
        cols = corrmat.nlargest(k, target)[target].index
        cm = np.corrcoef(df[cols].values.T)
        sns.set(font_scale=1.25)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                    yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

    return corrmat


def fill_cat_feature(df, cat_feature_list, fill_value):
    if fill_value == 'mode':
        for cat_feature in cat_feature_list:
            df[cat_feature].fillna(cat_feature.mode()[0], inplace=True)
    else:
        for cat_feature in cat_feature_list:
            df[cat_feature].fillna(value=fill_value, inplace=True)

    return df


def fill_num_feature(df, num_feature_list, fill_type):
    if fill_type == 'median':
        for num_feature in num_feature_list:
            df[num_feature].fillna(df[num_feature].median(), inplace=True)
    elif fill_type == 'mean':
        for num_feature in num_feature_list:
            df[num_feature].fillna(df[num_feature].mean(), inplace=True)
    elif fill_type == '0':
        for num_feature in num_feature_list:
            df[num_feature].fillna(0, inplace=True)

    return df


# put the target variable on the log1 scale

def log_target(target):
    log_y = np.log1p(target)  # using log1p to avoid cased of log0
    # check the plot
    sns.distplot(log_y, fit=norm)
    return log_y


def create_dummies(df):
    all_dummy = pd.get_dummies(df)
    # Check if all objects get into dummy variables
    all_dummy_col = all_dummy.dtypes[all_dummy.dtypes == 'int64'].index.sort_values()
    num_feature_list = df.dtypes[df.dtypes == 'int64'].index.sort_values()
    test_var = (all_dummy_col == num_feature_list).all()
    print('Does the dummy variables created in a right way?', test_var)
    print('After creating dummy variables, the total number of features are: ', all_dummy.shape)
    return all_dummy


# Model selection Tryout

def model_tryout(model,X,y,n_folds):
    scores = np.zeros((10,4))
    i= 0
    models = make_pipeline(RobustScaler(),model)
    kf = KFold(n_splits= n_folds, shuffle=True, random_state=42)#.get_n_splits(X)
    for train_index, test_index in kf.split(X, y):
        X_train,X_test,y_train,y_test = X.iloc[train_index],X.iloc[test_index],y.iloc[train_index],y.iloc[test_index]
        instance = models.fit(X_train.values,y_train)
        ypred_test = instance.predict(X_test) #Using test dataset to predict
        ypred_train = instance.predict(X_train) #Using train dataset to predict
        scores[i,0] = round(np.sqrt(mean_squared_error(y_test,ypred_test)),2) #rmse for cv scores
        scores[i,1] = round(np.sqrt(mean_squared_error(y_train,ypred_train)),2) #rmse for test
        scores[i,2] = round(r2_score(y_test,ypred_test),2)
        scores[i,3] = round(r2_score(y_train,ypred_train),2)
        i = i+1
    model_select = pd.DataFrame(scores,columns =['rmse_test','rmse_train','r2_test','r2_train'])
    return model_select

def overfit_plot(df,name):
        plt.figure()
        plt.plot(df['rmse_test'],c = 'r')
        plt.plot(df['rmse_train'],c = 'blue')
        plt.yticks(np.arange(0,.4,.05))
        plt.xlabel('Number of cv')
        plt.ylabel('RMSE')
        plt.legend(loc = 'best')
        plt.title('RMSE for {}'.format(name))
        plt.show()


    print('---------------------------------------------------')
    print('For the {} default model, the average RMSE for CV is: {:.2f}'.format(m_name, np.mean(rmse_cv)))
    print('For the {} default model, the average RMSE for the model is: {:.2f}'.format(m_name, rmse_model))
    print('For the {} default model, the average R2 for CV is: {:.2f}'.format(m_name, np.mean(r2_cv)))
    print('For the {} default model, the average R2 for model is: {:.2f}'.format(m_name, r2_model))
    return(round(np.mean(rmse_cv),2),round(rmse_model,2),round(np.mean(r2_cv),2),round(r2_model))


if __name__ == '__main__':
    # 1. Read in train and test data
    # 2. EDA - Data Examination
    # 2.1 & Split target and features
    train, test, train_id, test_id, train_y, train_x, df_all = importing_data()

    # 2.2 Checking for Missing Values
    na_stats = check_missing(df_all)

    # 2.3 Check for distribution of the target: y
    check_distribution(train, ['SalePrice'])

    # 2.4.1 Check for numeric variable correlations
    #     corrmat_num = check_correlations(train, check_type='num_feature', target='SalePrice')
    # 2.4.2 Check correlations for target
    corrmat_tar = check_correlations(train, check_type='target', target='SalePrice')

    # 3.Data Preprocessing
    # 3.1 Change the MSSubClass variable into categorical
    df_all['MSSubClass'] = df_all.MSSubClass.astype('object')

    # 3.2 Impute Missing values
    # 3.2.1 Based on the data description, the following features can be filled as "None" when they are missing
    miss_cat_feature_list_par1 = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond',
                                  'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure',
                                  'BsmtQual', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrType']
    df_all = fill_cat_feature(df_all, miss_cat_feature_list_par1, 'None')

    # 3.2.2 Categorical features but should be filled with something else instead of None
    # MSZoning, Utilities, Electrical filled with referring to neighborhood
    df_all.loc[(df_all['Neighborhood'] == 'IDOTRR') & (df_all.MSZoning.isnull()), 'MSZoning'] = 'RM'
    df_all.loc[(df_all['Neighborhood'] == 'Mitchel') & (df_all.MSZoning.isnull()), 'MSZoning'] = 'RL'
    df_all.Utilities.fillna('AllPub', inplace=True)
    df_all.Electrical.fillna('SBrkr', inplace=True)

    # BsmtHalfBath,BsmtFullBath referring to Neighborhood and BldgType
    df_all.BsmtHalfBath.fillna(0, inplace=True)
    df_all.BsmtFullBath.fillna(0, inplace=True)

    # Functional: filled with mode
    df_all.Functional.fillna(df_all.Functional.mode()[0], inplace=True)
    df_all.Exterior2nd.fillna('Other', inplace=True)  # Since unknown, filled with 'Other'
    df_all.Exterior1st.fillna('Other', inplace=True)
    df_all.SaleType.fillna('Oth', inplace=True)

    # KitchenQual filled by referring to other rooms' quality
    df_all.KitchenQual.fillna('TA', inplace=True)

    # 3.3.3 Numeric features that should be filled based on the distribution of the feature itself
    miss_num_feature_list_median = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    df_all = fill_num_feature(df_all, miss_num_feature_list_median, 'median')
    miss_num_feature_list_mean = ['LotFrontage']
    df_all = fill_num_feature(df_all, miss_num_feature_list_mean, 'mean')
    miss_num_feature_list_zero = ['GarageArea', 'GarageCars', 'MasVnrArea']
    df_all = fill_num_feature(df_all, miss_num_feature_list_zero, '0')

    # 4. Feature Engineering
    # 4.1 Drop features that have high correlations
    df_all.drop(['TotalBsmtSF', 'GarageCars', 'GarageYrBlt'], axis=1, inplace=True)
    # Checking for Missing Values
    na_stats_unmissing = check_missing(df_all)
    # 4.2 Get all categorical variables into dummy variables
    df_all = create_dummies(df_all)
    # 4.3 log target and add to df_train
    ntrain = train.shape[0]
    ntest = test.shape[0]
    df_train = df_all[:ntrain]
    df_test = df_all[ntrain:]
    print('---------------------------')
    print('Shape of train :', train.shape)
    print('Shape of df_train :', df_train.shape)
    print('Shape of test :', test.shape)
    print('Shape of df_test :', df_test.shape)
    print('---------------------------')
    train_y = log_target(train_y)
    #df_train = pd.concat([df_train, train['SalePrice_log']], axis=1)
    print('---------------------------')
    print('Shape of df_train after add target :', df_train.shape)
    print('---------------------------')

    # 5. Building models
    '''The models can be used to train data:
            1. Linear Regression - The most simple model
            2. Linear Regression with regularization parameters:
                2.1 Lasso: Based on Linear Regression, but with penalty of adding absolute values of coefficients, which
                can be used to select features
                2.2 Ridge: with the penaly by adding square values of coefficients
            3. Random Forest
            4. Boosting methods: AdaBoost, Gradient Descent Boost, XGboost'''

    from sklearn.linear_model import Lasso, ElasticNet,Ridge
    from sklearn.preprocessing import RobustScaler
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    import lightgbm as lgb

    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score

    # Set up a pipeline for lasso,ElasticNet
    lasso = model_tryout(Lasso(alpha=.0005, random_state=1), df_train, train_y, 10)
    ridge = model_tryout(Ridge(alpha = .0005, random_state = 1),df_train,train_y,10)
    Enet = model_tryout(ElasticNet(alpha=.0005, l1_ratio=9, random_state=1), df_train, train_y, 10)
    rf = model_tryout(RandomForestRegressor(verbose=False, random_state=1), df_train, train_y, 10)
    model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=10,
                                  learning_rate=0.05, n_estimators=720,
                                  max_bin=55, bagging_fraction=0.8,
                                  bagging_freq=5, feature_fraction=0.2319,
                                  feature_fraction_seed=9, bagging_seed=9,
                                  min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
    lgb = model_tryout(model_lgb, df_train, train_y, 10)

    # Model Evaluation
    '''
    model_compare = pd.DataFrame({'Model': ['Lasso', 'ElasticNet', 'RF', 'LGB'],
                                  'R_square_cv': [lasso[2], Enet[2], rf[2], lgb[2]],
                                  'R_square_model':[lasso[3], Enet[3], rf[3], lgb[3]],
                                  'RMSE_cv': [lasso[0], Enet[0], rf[0], lgb[0]],
                                  'RMSE_model':[lasso[1], Enet[1], rf[1], lgb[1]]}).sort_values(['R_square_cv'],
                                                                                           ascending=False)'''
    lasso_plot = overfit_plot(lasso, 'Lasso')
    ridge_plot = overfit_plot(ridge, 'Ridge')
    Enet_plot = overfit_plot(Enet, 'Enet')
    rf_plot = overfit_plot(rf, 'RandomForest')
    lgb_plot = overfit_plot(lgb, 'LGB')

    #Compare the mean rmse and r2 within and between models
    model_comp = pd.DataFrame()
    for key, value in {'Lasso': lasso, 'Ridge': ridge, 'Enet': Enet, 'RF': rf, 'LGB': lgb}.items():
        mean_evaluation = value.mean(axis=0)
        mean_df = pd.DataFrame(mean_evaluation).T
        model_comp = model_comp.append(mean_df.rename(index={0: key}))
    model_comp = model_comp.sort_values(['rmse_test', 'r2_test'], ascending=False)
