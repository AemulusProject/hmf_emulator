#Test file for extending the Aemulator base class.

from Aemulator import *

class myobj(Aemulator):

    def __init(self):
        Aemulator.__init__(self)
        print("i exist")

if __name__=="__main__":
    m = myobj()
    print(m)
