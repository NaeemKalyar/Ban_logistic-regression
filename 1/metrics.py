"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""


def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    Y_true = y_true.to_list()
    
    right = 0
    for i in range(len(Y_true)):
        if Y_true[i] == y_pred[i]:
            right += 1
    return right / int(len(Y_true)) * 100.0

    

def precision_score(y_true, y_pred):
    
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    Y_true = y_true.to_list()
    tpositive = 0
    fpositive = 0
     
    for act, pred in zip(Y_true, y_pred):
        if act == 1 and pred == 1:
            tpositive +=1
        if act==0 and pred ==1:
            fpositive +=1
            
    return tpositive/(tpositive + fpositive)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    Y_true = y_true.to_list()
    tpositive = 0
    fnegative = 0
     
    for act, pred in zip(Y_true, y_pred):
        if act == 1 and pred == 1:
            tpositive +=1
        if act==1 and pred ==0:
            fnegative +=1
            
    return tpositive/(tpositive + fnegative)

    


def f1_score(y_true, y_pred):
    
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    #Y_true = y_true.to_list()
    p=precision_score(y_true,y_pred)
    r=recall_score(y_true,y_pred)
    f1=2*p*r/(p+r)
    return f1