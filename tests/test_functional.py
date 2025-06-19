import os
import unittest
import pickle
import numpy as np

from code.ml import Model

#file_path = file_path = os.path.dirname(os.path.realpath(__file__)) + '/../output_revised.txt'

try:
    model = Model()
    X_train, X_test, y_train, y_test = model.data_transformation()
except:
    pass
from tests.TestUtils import TestUtils
class FuctionalTests(unittest.TestCase):
    def test_train_test_split_reproduciability(self):
        test_obj = TestUtils()
        try:
            X_train2, X_test2, y_train2, y_test2 = model.data_transformation()

            if (X_test.equals(X_test2)):
                passed = True
                test_obj.yakshaAssert("TestTrainTestSplitReproduciability", True, "functional")
                print("TestTrainTestSplitReproduciability = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestTrainTestSplitReproduciability", False, "functional")
                print("TestTrainTestSplitReproduciability = Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestTrainTestSplitReproduciability", False, "functional")
            print("TestTrainTestSplitReproduciability = Failed")
        assert passed

    def test_train_vs_test_features(self):
        test_obj = TestUtils()
        try:
            if (X_test.shape[1] == X_train.shape[1]):
                passed = True
                test_obj.yakshaAssert("TestTrainVsTestFeartures", True, "functional")
                print("TestTrainVsTestFeartures = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestTrainVsTestFeartures", False, "functional")
                print("TestTrainVsTestFeartures = Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestTrainVsTestFeartures", False, "functional")
            print("TestTrainVsTestFeartures = Failed")
        assert passed

    def test_numerical_features_imputed(self):
        test_obj = TestUtils()
        try:
            na_found_train = X_train.isnull().any().any()
            na_found_test = X_test.isnull().any().any()

            if not (na_found_train or na_found_test):
                passed = True
                test_obj.yakshaAssert("TestNumericalFeaturesImputed", True, "functional")
                print("TestNumericalFeaturesImputed = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestNumericalFeaturesImputed", False, "functional")
                print("TestNumericalFeaturesImputed = Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestNumericalFeaturesImputed", False, "functional")
            print("TestNumericalFeaturesImputed = Failed")
        assert passed

    def test_pipeline_custom_model_random_state(self):
        test_obj = TestUtils()
        try:
            loaded_model = pickle.load(open(model.model_file, 'rb'))
            model_random_state = loaded_model.random_state
            if model_random_state is not None:
                passed = True
                test_obj.yakshaAssert("TestPipelineCustomModelRandomState", True, "functional")
                print("TestPipelineCustomModelRandomState = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestPipelineCustomModelRandomState", False, "functional")
                print("TestPipelineCustomModelRandomState = Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestPipelineCustomModelRandomState", False, "functional")
            print("TestPipelineCustomModelRandomState = Failed")
        assert passed
