#!/usr/bin/env python

# Copyright 2011 Sheng-Yao Tseng
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Learn, save model, and predict.

"""
__author__ =  'Sheng-Yao Tseng'
__version__=  '1.0'

from logilab.hmm import HMM
from numpy import array,eye,add
import getopt
import sys

DEBUG = False
showpra = False or DEBUG
dump = False or DEBUG

def computepra(T):
    """Each element of T, t_ab, should be the times that the observated state trans to b from a.  """
    
    if showpra:
        print """--------------------------------------------------------------------------------"""
        print """Originally input matrix."""
        print T
        print """--------------------------------------------------------------------------------"""

    R = T + 1

    if showpra:
        print """Smoothed input matrix."""
        print R
        print """--------------------------------------------------------------------------------"""
        print """Transport of smoothed input matrix."""

    RConj = R.conj().T

    if showpra:
        print RConj
        print """--------------------------------------------------------------------------------"""

    
    #########################################################################
    # Assumed the input matrix T is a 3x3 matrix, then
    #
    #     | a b c |
    # R = | d e f |, which is a 3x3 matrix,
    #     | g h i | 
    # 
    # and RowSum = |(a+b+c) (d+e+f) (g+h+i)| which is a vector with 3 elements.
    #
    ###########################################################################

    RowSum = add.reduce(R, 1)

    if showpra:
        print """Sum of rows of smoothed input matrix."""
        print RowSum
        print """--------------------------------------------------------------------------------"""
     
    #############################################################################
    # What is the exactly operations of A_{nxn} / B_{1xn} in numpy? It means:
    # [first_column of A] / B_1 and
    # [second_column of A] / B_2 and ... etc.
    # So I transported R (thus obtained RConj) and divide it by sum of row of R, 
    # creating a probability matrix.
    # Note that logilab.hmm.HMM required the probability matrix should be a 
    # row-probability matrix , so I transported it back.
    # 
    # Be uncertain, please refer the numpy manual. The links of online version follows:
    # 
    #   http://www.scipy.org/Numpy_Example_List
    #   http://www.scipy.org/NumPy_for_Matlab_Users
    #
    ###############################################################################
    Temp = RConj / RowSum
    statetransition = Temp.conj().T

    if showpra: 
        print """The final state transition matrix."""
        print statetransition
        print """--------------------------------------------------------------------------------"""
        print """Sum up the rows ..."""
        print add.reduce(statetransition,1)
        print """And minus each elements by 1."""
        print add.reduce(statetransition,1) - 1
        print """--------------------------------------------------------------------------------"""

def usage():
    """usage function."""
    print """
USAGE: computeHMM.py command options ...
    
    Command:

    display - Load a model and display its contents.
    judge   - computed the next result of observation
    train   - Train model using given dataset file.

    Example:
              ex: 
              computeHMM.py train dataset_file model_file   
              computeHMM.py judge model_file observation
              computeHMM.py display model_file
              """

def checkEmptyFN(trainFN=None, modelFN=None, observation=None):
    """Check if any filename is empty. I wrote it to prevent annoying exception messages."""
    if trainFN == '':
        print """Please specifiy a file with training sequences. Stop."""
        sys.exit(1)

    if modelFN == '':
        print """Please specifiy the name of model file. Stop."""
        sys.exit(1)

    if observation == '':
        print """No observation file and can't judge anything. Stop."""
        sys.exit(1)

def doTrain(trainFN, modelFN):
    """Load trainging data from file named in trainFN and store the model
    into file named in modelFN

    Keyword arguments:
        trainFN -- File name of training data file.
        modelFN -- File name to the file where HMM model stored in."""

    checkEmptyFN(trainFN=trainFN, modelFN=modelFN)
    
    # This is the statistic from test data; hardcoded.
    T = array([[5., 2., 2., 34.],
              [0., 2., 1., 26.],
              [2., 2., 4., 48.],
              [34., 23., 47., 1013.]])

    statetransition = computepra(T)

    detector = HMM(['Temporal', 'Contingency', 'Comparison', 'Expansion'],
            ['1','2','3','4'],
            transition_proba=statetransition, 
            observation_proba=eye(4),
            initial_state_proba=[.25, .25, .25, .25])

    observed = []

    try:
        with open(trainFN, "r") as S:
            observed = [l[:-1] for l in S.readlines()]
    except IOError as (errno, strerror):
            print "I/O error({0}): {1}".format(errno, strerror)
    except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

    detector.multiple_learn(observed, maxiter=2000)

    if dump:
        detector.dump()

    try:
        detector.saveHMM(open(modelFN,"w"),saveState=True)
    except IOError as (errno, strerror):
        print "I/O error({0}): {1}".format(errno, strerror)
    except:
        print "Unexpected error:", sys.exec_info()[0]
        raise

def doJudge(modelFN, observationFN):
    """Load model and observation from modelFN and observationFN then make prediction

    Keyword arguments:
        modelFN -- File name to the file where HMM model stored in.
        observationFN -- The name of file that observation sequences stored in."""

    checkEmptyFN(modelFN=modelFN, observation=observationFN)

    h = HMM(['a'],['1'])    # A trivial HMM model.
    obs = ''

    try:
        with open(modelFN, "r") as M:
            h.loadHMM(M)
            if dump:
                h.dump()
    except IOError as (errno, strerror):
          print "I/O error({0}): {1}".format(errno, strerror)
    except:
          print "Unexpected error:", sys.exec_info()[0]
          raise

    try:
        with open(observationFN, "r") as M:
            obs = ''.join([l.strip('\r\n') for l in M])
    except IOError as (errno, strerror):
          print "I/O error({0}): {1}".format(errno, strerror)
    except:
          print "Unexpected error:", sys.exec_info()[0]
          raise

    # Note that HMM.omega_X is "State name" and omega_O is "output state"!!

    O = h.analyze(obs)
    i = h.omega_X.index(O[-1])
    print O[-1]

#    col = h.A[:,i:i+1]
#    zipped = zip(h.omega_O, col)
#    output, prob = max(zipped, key=lambda x : x[1])
#    print "{0}:{1}".format(output, prob[0])
    
    row = h.A[i]
    print row

    zipped = zip(h.omega_O, row)
    output, prob = max(zipped, key=lambda x: x[1])
    print "{0}:{1}".format(output, prob)



def doDisplay(modelFN):
    """Display the content of HMM model stored in modelFN."""

    h = HMM(['a'],['1'])    # A trivial HMM model.
    try:
        with open(modelFN, "r") as M:
            h.loadHMM(M)
            h.dump()
            print ""
            print h.omega_X
            print h.omega_O
    except IOError as (errno, strerror):
          print "I/O error({0}): {1}".format(errno, strerror)
    except:
          print "Unexpected error:", sys.exec_info()[0]
          raise

def main():
    global showpra, dump, DEBUG
    
    training = '' 
    model = '' 
    observation = ''

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    else:
        command = sys.argv[1]

    if command == 'train':
        training, model = sys.argv[2:4]
        doTrain(training, model)
    elif command == 'judge':
        model, observation = sys.argv[2:4]
        doJudge(model, observation)
    elif command == 'display':
        model = sys.argv[2]
        doDisplay(model)
    else:
        usage()
        sys.exit(1)


if __name__ == '__main__':
    main()

