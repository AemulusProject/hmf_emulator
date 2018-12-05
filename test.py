#Test file for extending the Aemulator base class.

from Aemulator import *

class hmf_emulator(Aemulator):

    def __init(self):
        Aemulator.__init__(self)
        print("i exist")

    def load_data(self, filename):
        pass

    def build_emulator(self, hyperparams=None):
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
