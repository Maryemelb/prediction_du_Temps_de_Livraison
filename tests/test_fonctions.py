import pytest
import sys
import os

# Add the project root (one level up from "tests") to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from functions.fonctions import data_preprocessing,display_metrics

def test_verifier_format_dim():
    X_train, X_test, y_train, y_test,X,y= data_preprocessing()
    # assert y.dtype == 'int64'
    # assert X['Time_of_Day'].dtype == 'object'
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0]== y_test.shape[0]
def test_gridsearch_metrics_mae():
    mae_rf, mae_svr= display_metrics()
    assert mae_rf <=8    
    assert mae_svr <=8