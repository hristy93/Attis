from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import statistics as s
import math

def show_cross_validation_score(classificator, X, y):
    """ Shows the 10-fold cross validation scores and their 
        average using the classificator and some partial (X) 
        and target (y) values 
    """
    k_fold_count = 10
    scores = cross_val_score(classificator, X, y, cv=k_fold_count, n_jobs=-1)
    print("  {0}-fold cross validation scores: {1}".format(k_fold_count, scores))
    print("  Average score: {0}".format(s.mean(scores)))

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

def test_decision_tree_classification(X_train, X_test, y_train, y_test):
    """ Tests a decision tree classification with some training 
        (X_train, y_train) and testing (X_test, y_test) data 
    """
    print("\nTesting DecisionTreeClassifier ...")
    dtc_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
        max_depth=3, min_samples_leaf=5)
    dtc_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = accuracy_score(y_test, y_pred)
    print("Accuracy_score: ", accuracy_score(y_test, y_pred))
    print("Classification_report:\n", classification_report(y_test, y_pred))
    print(result)

def test_decision_tree_classification_with_cv(X, y):
    """ Tests a decision tree classification with partial (X) and 
        target (y) values using cross validation
    """
    print("\nTesting DecisionTreeClassifier with cross valdiation ...")
    dtc_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
        max_depth=2, min_samples_leaf=2)
    show_cross_validation_score(dtc_entropy, X, y)

def test_decision_tree_regression(X_train, X_test, y_train, y_test, max_depth=2):
    """ Tests a decision tree regression with some training 
        (X_train, y_train) and testing (X_test, y_test) data 
    """
    print("\nTesting DecisionTreeRegressor ...")
    dtr_entropy = DecisionTreeRegressor(max_depth=max_depth)
    dtr_entropy.fit(X_train, y_train)
    y_pred = clf_entropy.predict(X_test)
    result = math.sqrt(mean_squared_error(y_test, y_pred))
    print(result)

def test_decision_tree_regression_with_cv(X, y, max_depth=2):
    """ Tests a decision tree regression with partial (X) and 
        target (y) values using cross validation
    """
    print("\nTesting DecisionTreeRegressor with cross valdiation ...")
    dtr_entropy = DecisionTreeRegressor(max_depth=max_depth)
    show_cross_validation_score(dtr_entropy, X, y)

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

def association_rules_test(dataframe, support=0.6):
    """ Tests association rules with support """
    print("\nTesting asocciation rules ...")
    frequent_itemsets = apriori(dataframe, min_support=support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    print("  Frequent itemsets:")
    print(frequent_itemsets)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print("  Association rules:")
    print(rules.head())
    refined_frequent_itemsets = frequent_itemsets[(frequent_itemsets['length'] == 2) &
                                (frequent_itemsets['support'] >= 0.8) ]
    print("  Refined frequent itemsets:")
    print(refined_frequent_itemsets)
    refined_rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    print("  Refined association rules:")
    print(refined_rules.head())


