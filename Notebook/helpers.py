
def missing_data(df, level = 5):
    """
    Description:
    Inputs:
    Output:
    """
    try:
        missings = (df.isnull().sum()/len(df)*100).sort_values(ascending=False)
        missings = pd.DataFrame({'Feature':missings.index, 'Total NaN':missings.values})
        missings[missings['Total NaN'] > level].plot.bar(x='Feature', y='Total NaN', figsize=(15, 5), color = 'blue')
        display(missings[missings['Total NaN'] > level]['Feature'])
        return missings
    except:
        print("Not feature with more than {}% NaNs".format(level))
        
def drop_missings(df, missings, thresold = 5):
    """
    Description:
    Inputs:
    Output:
    """
    outliers = missings[missings["Total NaN"] >=  thresold]['Feature']
    display(outliers)
    df.drop(outliers, axis=1, inplace=True)
    
def convert_to_number(df, col, sign):
    return df[col].str.strip(sign).str.replace(',', '').astype(float)



def removing_outliers(df, columns, q1 = 0.25, q2 = 0.74):
    Q1 = df[columns].quantile(q1)
    Q3 = df[columns].quantile(q2)
    IQR = Q3 - Q1
    df_without_outliers = df[~((df[columns] < (Q1 - 1.5 * IQR)) |
                               (df[columns] > (Q3 + 1.5 * IQR)))
                            ]
    return df_without_outliers



def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    

    
def get_components(components, columns, n, abs_opt = False):
    
    if abs_opt:
        for col in pca_comps.columns:
            pca_comps = pd.DataFrame(components[0:n].T, index = columns, columns = ["PCA " + str(i + 1) for i in range(n)])
            pca_comps[col] = abs(pca_comps[col].values)
    else:
            pca_comps = pd.DataFrame(components[0:n].T, index = columns, columns = ["PCA " + str(i + 1) for i in range(n)])
    
    return pca_comps


def feature_plot(importances, X_train, y_train, n = 5):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:n]]
    values = importances[indices][:n]

    # Creat the plot
    fig = plt.figure(figsize = (10,5))
    plt.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    plt.bar(np.arange(n), values, width = 0.6, align="center", color = 'blue', \
          label = "Feature Weight")
    plt.bar(np.arange(n) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = 'navy', \
          label = "Cumulative Feature Weight")
    plt.xticks(np.arange(n), columns)
    plt.xlim((-0.5, n-0.5))
    plt.ylabel("Weight", fontsize = 12)
    plt.xlabel("Feature", fontsize = 12)
    plt.xticks(rotation = 90);
    
    plt.legend(loc = 'upper center')
    plt.tight_layout()
    plt.show() 
    
def model_fit(df,  city, target, outlier = False, 
              remove_features = False, classifier = False
             ):
    
    try:
        df = df[df['city_'+city]==1]
    except:
        print('Check city name')

    if outlier:
        df = removing_outliers(df, outlier)
    if remove_features:
        df = df.loc[:, df.columns != remove_features]

    scaler = RobustScaler()
    df_std = pd.DataFrame(scaler.fit_transform(df), 
                              columns = df.columns)
    y = df_std[target]
    X = df_std.loc[:, df_std.columns != target]
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Split the 'features' and 'income' data into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size = 0.2, 
                                                        random_state = 0)


    if classifier:
        model_XGB = XGBClassifier(random_state=1) 
        model_XGB.fit(X_train,y_train)
        preds_XGB = model_XGB.predict(X_test)
        accuracy = accuracy_score(y_test, preds_XGB)
        print("Accuracy: %f" % (accuracy)) 
    else:
        model_XGB = XGBRegressor(random_state=1) 
        model_XGB.fit(X_train,y_train)
        preds_XGB = model_XGB.predict(X_test)
        rmse_XGB = np.sqrt(mean_squared_error(y_test, preds_XGB))
        print("RMSE: %f" % (rmse_XGB))
        
    importances = model_XGB.feature_importances_
    feature_plot(importances, X_train, y_train)
    
def ploting_main_pca_comp(n = 5):
    base_color = sb.color_palette("Paired")[1]
    rows = 2
    cols = 2
    #ax, fig = plt.subplots(rows, cols)
    df_pca = pd.DataFrame(pca_boston["PCA 1"].sort_values(ascending=False).head(n))

    plt.subplot(rows, cols, 1)
    column = df_pca.columns
    plt.title("Boston positive associate attribute")
    sb.barplot(data = df_pca, x = df_pca.index.values, y = "PCA 1", color = base_color);
    plt.xticks(rotation = 90);

    df_pca = pd.DataFrame(pca_boston["PCA 1"].sort_values(ascending=True).head(n))

    plt.subplot(rows, cols, 2)
    column = df_pca.columns
    plt.title("Boston negative associate attribute")
    sb.barplot(data = df_pca, x = df_pca.index.values, y = "PCA 1", color = base_color);
    plt.xticks(rotation = 90);
    plt.tight_layout();


    df_pca = pd.DataFrame(pca_seattle["PCA 1"].sort_values(ascending=False).head(n))
    plt.subplot(rows, cols, 3)
    column = df_pca.columns
    plt.title("Seattle positive associate attribute")
    sb.barplot(data = df_pca, x = df_pca.index.values, y = "PCA 1", color = base_color);
    plt.xticks(rotation = 90);

    df_pca = pd.DataFrame(pca_seattle["PCA 1"].sort_values(ascending=True).head(n))
    plt.subplot(rows, cols, 4)
    column = df_pca.columns
    plt.title("Seattle negative associate attribute")
    sb.barplot(data = df_pca, x = df_pca.index.values, y = "PCA 1", color = base_color);
    plt.xticks(rotation = 90);

    plt.show();