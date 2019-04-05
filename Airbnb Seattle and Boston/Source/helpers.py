import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats
from xgboost import XGBRegressor
from xgboost import XGBClassifier

import os


def missing_data(df, level=5):
    """
    Process missing data 

    INPUT: df - DataFrame to Process
           level = Aceptable %level of NaN

    OUTPUT:
            missings = df with attributes and its % NaN
    """
    try:
        missings = (df.isnull().sum() / len(df) * 100).sort_values(
            ascending=False)
        missings = pd.DataFrame({
            'Feature': missings.index,
            'Total NaN': missings.values
        })
        sb.set(style="white")
        sb.set_color_codes("pastel")
        sb.despine()
        missings[missings['Total NaN'] > level].plot.bar(
            x='Feature', y='Total NaN', figsize=(15, 5), color='b')
        display(missings[missings['Total NaN'] > level]['Feature'])
        return missings
    except:
        print("Not feature with more than {}% NaNs".format(level))


def drop_missings(df, missings, thresold=5):
    """
    Drop missing data 

    INPUT: df - DataFrame to Process
           missing = df with attributes and its % NaN
           thresold= Aceptable %level of NaN

    OUTPUT:
            None
    """
    outliers = missings[missings["Total NaN"] >= thresold]['Feature']
    display(outliers)
    df.drop(outliers, axis=1, inplace=True)


def convert_to_number(df, col, sign):
    """
    Convert columns with special signs to number

    INPUT: df - DataFrame to Process
           col - Col to process

    OUTPUT:
            df proceseed
    """
    return df[col].str.strip(sign).str.replace(',', '').astype(float)


def removing_outliers(df, columns):
    """
    Remove outliers of columns

    INPUT: df - DataFrame to Process
           col - Col to process

    OUTPUT:
            df proceseed
    """
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    df_without_outliers = df[~((df[columns] < (Q1 - 1.5 * IQR)) |
                               (df[columns] > (Q3 + 1.5 * IQR)))]
    return df_without_outliers


def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 

    INPUT: pca - the result of instantian of PCA in scikit learn

    OUTPUT:
            None
    '''
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    print(cumvals)
    ax.plot(ind, cumvals)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


def get_components(components, columns, n, abs_opt=False):
    """
    Get n PCA 

    INPUT: components - pcas
           columns - columns of components
           n - numbers of componnents

    OUTPUT:
            pca_comps - n PCA components
    """

    if abs_opt:
        for col in pca_comps.columns:
            pca_comps = pd.DataFrame(
                components[0:n].T,
                index=columns,
                columns=["PCA " + str(i + 1) for i in range(n)])
            pca_comps[col] = abs(pca_comps[col].values)
    else:
        pca_comps = pd.DataFrame(
            components[0:n].T,
            index=columns,
            columns=["PCA " + str(i + 1) for i in range(n)])

    return pca_comps


def feature_plot(importances, X_train, y_train, n=5):
    """
    plot n importante feature of a model

    INPUT: importances - df with features importances
           X_train - Matrix with X train values
           y_train - vector with y values
           n - numbers of features

    OUTPUT:
            None
    """

    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n]]
    values = importances[indices][:n] * 100

    # Creat the plot
    plt.figure(figsize=(5, 3))
    sb.set(style="white")
    sb.set_color_codes("pastel")
    sb.barplot(x=values, y=columns, label="Weight (%)", color="b")
    sb.despine()

    plt.legend()


def model_fit(df,
              city,
              target,
              outlier=False,
              remove_features=False,
              classifier=False):
    """
    plot n importante feature of a model

    INPUT: df - DataFrame
           city - city to subset
           target - target variable
           outlier - True remove outlier
           remove_features - True remove list of feature
           classifier - True is a classifier model

    OUTPUT:
            None
    """
    try:
        df = df[df['city_' + city] == 1]
    except:
        print('Check city name')

    if outlier:
        df = removing_outliers(df, outlier)
    if type(remove_features) is not bool:
        df = df.drop(remove_features, axis=1)

    scaler = RobustScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    y = df_std[target]
    X = df_std.loc[:, df_std.columns != target]
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Split the 'features' and 'income' data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    if classifier:
        model_XGB = XGBClassifier(random_state=1)
        model_XGB.fit(X_train, y_train)
        preds_XGB = model_XGB.predict(X_test)
        accuracy = accuracy_score(y_test, preds_XGB)
        print("Accuracy: %f" % (accuracy))
    else:
        model_XGB = XGBRegressor(random_state=1)
        model_XGB.fit(X_train, y_train)
        preds_XGB = model_XGB.predict(X_test)
        rmse_XGB = np.sqrt(mean_squared_error(y_test, preds_XGB))
        r2 = r2_score(y_test, preds_XGB)
        print("RMSE: %f\nR2: %f" % (rmse_XGB, r2))

    importances = model_XGB.feature_importances_
    feature_plot(importances, X_train, y_train)

    # plot side by side PCA_comps


def violinplot_multi(df, x, y, hue):
    """
    plot multivariate violinplot

    INPUT:  df - DataFrame
            x - x
            y - y
            hue - hue

    OUTPUT:
            None
    """

    sb.set(style="white", palette="pastel", color_codes=True)
    palette = [sb.color_palette("Paired")[1], sb.color_palette("Paired")[0]]
    opacity = 0.8
    # Draw a nested violinplot and split the violins for easier comparison
    sb.violinplot(x=x, y=y, hue=hue,
                   split=True, inner=None,alpha=opacity,
                   palette={"Boston": palette[0], "Seattle": palette[1]},
                   data=df, cut = 0)
    sb.despine(left=True)


def violin_neighbourhood(df, x, y, city, figsize=(8, 10)):
    """
    plot multivariate violin_neighbourhood

    INPUT:  df - DataFrame
            x - x
            y - y
            city - city to subset
            hue - hue

    OUTPUT:
            None
    """

    order = df[df['city'] == city].groupby(
        'neighbourhood_cleansed')[x].min().sort_values(ascending=True).index
    opacity = 0.8
    plt.figure(figsize=figsize)
    sb.set(style="white", palette="pastel", color_codes=True)
    base_color = sb.color_palette("Paired")[1]
    # Draw a nested violinplot and split the violins for easier comparison
    sb.violinplot(
        x=x,
        y=y,
        inner=None,
        alpha=opacity,
        data=df[df['city'] == city],
        cut=0,
        color=base_color,
        order=order)

    sb.despine()
    plt.xticks(rotation=90)