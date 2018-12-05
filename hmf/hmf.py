#Test file for extending the Aemulator base class.

from Aemulator import *
import numpy as np
import os, inspect

class hmf_emulator(Aemulator):

    def __init__(self):
        Aemulator.__init__(self)
        self.loaded_data = False
        self.load_data()

    def load_data(self, path_to_training_data_directory = None):
        """
        Load training data directly from file, and attach it to this object. 
        This method does not need to be called by the user.
        :param path_to_training_data_directory:
            Location of the training data. Must be in .npy format.
        :return:
            None
        """
        if path_to_training_data_directory is None:
            #Determine the local path to the data files folder
            data_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1])) + "/data_files/"
        else:
            data_path = path_to_training_data_directory
        self.data_path = data_path
            
        #Load all training data
        self.training_cosmologies = \
            np.load(data_path+"training_cosmologies.npy")
        self.rotation_matrix      = \
            np.load(data_path+"rotation_matrix.npy")
        self.training_data        = \
            np.load(data_path+"rotated_MF_parameters.npy")
        self.training_mean   = self.training_data[:,:,0] #sample means
        self.training_stddev = self.training_data[:,:,1] #sample stddevs
        self.loaded_data = True
        return

    def build_emulator(self, hyperparams=None):
        """
        Build the emulator directly from loaded training data.
        Optionally provide hyperparameters, 
        if something other than the default is preferred.
        :param hyperparams:
            A dictionary of hyperparameters for the emulator. Default is None.
        :return:
            None
        """
        if not self.loaded_data:
            raise Exception("Need to load training data first.")
        pass

    def train_emulator(self):
        pass

    def cache_emulator(self, filename):
        pass

    def load_emulator(self, filename):
        pass

    def predict(self, params):
        pass

if __name__=="__main__":
    e = hmf_emulator()
    print(e)
