#region Test functions for algorithms

def test_decision_tree_classification(X_train, X_test, y_train, y_test):
    """ Tests a decision tree classification with some training 
        (X_train, y_train) and testing (X_test, y_test) data 
    """
    print("\nTesting DecisionTreeClassifier ...")
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = accuracy_score(y_test, y_pred)
    print("accuracy_score: ", accuracy_score(y_test, y_pred))
    print("classification_report:\n", classification_report(y_test, y_pred))
    print(result)

def test_decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=2):
    """ Tests a decision tree regression with some training 
        (X_train, y_train) and testing (X_test, y_test) data 
    """
    print("\nTesting DecisionTreeRegressor ...")
    clf_entropy = DecisionTreeRegressor(max_depth=max_depth)
    clf_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = math.sqrt(mean_squared_error(y_test, y_pred))
    print(result)

def test_decision_tree_regression_with_cv(X, y, max_depth=2):
    """ Tests a decision tree regression with partial (X) and 
        target (y) values using cross validation
    """
    print("\nTesting DecisionTreeRegressor with cross valdiation ...")
    clf_entropy = DecisionTreeRegressor(max_depth=max_depth)
    show_cross_validation_score(clf_entropy, X, y)

def test_gradient_boosting_regression(X_train, X_test, y_train, y_test):
    """ Tests a gradient boosting regression with some some training 
        (X_train, y_train) and testing (X_test, y_test) data 
    """
    print("\nTesting GradientBoostingRegressor ...")
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    result = gbr.score(X_test, y_test)
    print("Score: ".format(result))

def test_gradient_boosting_regression_with_cv(X, y):
    """ Tests a gradient boosting regression with partial (X) and 
        target (y) values using cross validation
    """
    print("\nTesting GradientBoostingRegressor with cross valdiation ...")
    gbr = GradientBoostingRegressor()
    show_cross_validation_score(gbr, X, y)

def test_gradient_boosting_classification(X_train, X_test, y_train, y_test, max_depth=3):
    """ Tests a gradient boosting classification with some some training 
        (X_train, y_train) and testing (X_test, y_test) data 
    """
    print("\nTesting GradientBoostingClassifier ...")
    gbc = GradientBoostingClassifier(max_depth=max_depth)
    gbc.fit(X_train, y_train)
    result = gbc.score(X_test, y_test)
    print("Score: ".format(result))

def test_gradient_boosting_classification_with_cv(X, y, max_depth=3):
    """ Tests a gradient boosting classification with partial (X) and 
        target (y) values using cross validation
    """
    print("\nTesting GradientBoostingClassifier with cross valdiation ...")
    gbc = GradientBoostingClassifier(max_depth=max_depth)
    show_cross_validation_score(gbc, X, y)

def linear_regression_test_with_cv(X, y):
    """ Tests with liear regression with partial (X) and 
        target (y) values using cross validation
    """
    print("\nTesting LinearRegression with cross valdiation ...")
    regression_model = LinearRegression()
    show_cross_validation_score(regression_model, X, y)

def association_rules_test(dataframe, support):
    """ Tests association rules with support """
    print("\nTesting asocciation rules ...")
    frequent_itemsets = apriori(dataframe, min_support=support, use_colnames=True)
    print("  Frequent itemsets:")
    print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print("  Association rules:")
    print(rules.head())

#endregion

