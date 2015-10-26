###
# Test Suite for specified Attractor interface
#
# Run with the command: "nosetests test_attractor.py"
###

import unittest
from attractor import Attractor
from random import uniform, randint
from math import sqrt
from nose import with_setup

class TestEuler(unittest.TestCase):
    """This is set of basic test for Euler method"""
    def setUp(self):
        self.a = Attractor()


    def test_dataframe(self):
        """Make sure the data frame is created with right number of rows"""
        df = self.a.evolve(order=1)
        print "This is length of dataframe", len(df)
        assert len(df)-1 == 10000  # subrtact 1 for the title row
    
    def test_filesave(self):
        """Test ot make sure we are able to save the file"""
        file_save_status = self.a.save()
        print "The status of file save is ", file_save_status
        assert file_save_status == True

        
class Runga_Kutta(unittest.TestCase):
    """This is set of basic test for Euler method"""
    def setUp(self):
        self.a = Attractor()

    def test_dataframe(self):
        """Make sure the data frame is created with right number of rows"""
        df = self.a.evolve(order=4)
        print "This is length of dataframe", len(df)
        assert len(df)-1 == 10000  # subrtact 1 for the title row
    
    def test_filesave(self):
        """Test ot make sure we are able to save the file"""
        file_save_status = self.a.save()
        print "The status of file save is ", file_save_status
        assert file_save_status == True
        
    def test_diff_start_points(self):
        """Make sure the data frame is created when using non default values"""
        df = self.a.evolve(r0=[1.0,1.5,2.0],order=4)
        print " The first row of dataframe is ",  df.iloc[2,]
        print "This starting values of X variable is ", df.iloc[0,1]
       
        
def main():
    unittest.main()

if __name__ == '__main__':
    main()



