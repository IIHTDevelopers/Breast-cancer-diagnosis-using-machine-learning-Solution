#import os
import unittest
import numpy as np
# from code.ml import Model
from code import constants
#model = Model()
#file_path = os.path.dirname(os.path.realpath(__file__)) + '/../output_exception_revised.txt'
from tests.TestUtils import TestUtils
class ExceptionalTests(unittest.TestCase):
    def test_model_exists(self):
        test_obj = TestUtils()
        try:
            model_exists = os.path.exists(constants.MODEL_FOLDER + constants.MODEL_FILE)
            if model_exists:
                passed  = True
                test_obj.yakshaAssert("TestModelExists", True, "exception")
                print("TestModelExists = Passed")
            else:
                passed = False
                test_obj.yakshaAssert("TestModelExists", False, "exception")
                print("TestModelExists = Failed")
        except:
            passed = False
            test_obj.yakshaAssert("TestModelExists", False, "exception")
            print("TestModelExists = Failed")
        assert passed

    # def test_graphs_exists(self):
    #     test_obj = TestUtils()
    #     try:
    #         n_graphs = len(os.listdir(constants.GRAPHS_FOLDER))
    #         if n_graphs != 0:
    #             passed = True
    #             test_obj.yakshaAssert("TestGraphsExists", True, "exception")
    #             print("TestGraphsExists = Passed")
    #         else:
    #             passed = False
    #             test_obj.yakshaAssert("TestGraphsExists", False, "exception")
    #             print("TestGraphsExists = Failed")
    #     except:
    #         passed = False
    #         test_obj.yakshaAssert("TestGraphsExists", False, "exception")
    #         print("TestGraphsExists = Failed")
    #     assert passed
