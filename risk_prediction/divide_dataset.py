from sklearn.model_selection import train_test_split


def divide_dataset(all_features, all_targets):
    '''
    Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    return array/list
    '''
    x_train, y_train, x_test, y_test = train_test_split(all_features, all_targets, test_size=0.3, random_state=0)
    return x_train, y_train, x_test, y_test
