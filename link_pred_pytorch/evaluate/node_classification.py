######## node classification evaluation file ########
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, f1_score



def cls_evaluate_classifier(x_train, y_train, x_val, y_val, x_test, y_test, node_embeds):
    """ Downstream logistic regression classifier to evaluate node classification
        input: x:: numpy array of shape (num_samples, ), storing node ids (index from 0)
        input: y:: numpy array of shape (num_samples, ), storing node classes (0: positive; 1: negative)
    """

    # prepare necessary data and associated labels
    train_data = node_embeds[x_train]
    train_labels = y_train

    val_data = node_embeds[x_val]
    val_labels = y_val

    test_data = node_embeds[x_test]
    test_labels = y_test

    # train a linear model
    logistic = linear_model.LogisticRegression(class_weight='balanced')
    logistic.fit(train_data, train_labels)
    
    test_predict = logistic.predict_proba(test_data)[:, 1]
    val_predict = logistic.predict_proba(val_data)[:, 1]

    test_roc_score = roc_auc_score(test_labels, test_predict)
    val_roc_score = roc_auc_score(val_labels, val_predict)

    test_predict  = logistic.predict(test_data)
    val_predict   = logistic.predict(val_data)
    
    test_f1_score = f1_score(test_predict, test_labels)
    val_f1_score  = f1_score(val_predict,  val_labels)
    
    result = {
        'test_roc_score': test_roc_score,
        'val_roc_score':  val_roc_score,
        'test_predict':   test_predict,
        'val_predict':    val_predict,
        'test_f1_score':  test_f1_score,
        'val_f1_score':   val_f1_score,
    }

    return result

