import random as ran
import numpy as num
import keras

from randomgen import RandomGenerator,PCG64,ThreeFry,Philox,MT19937,Xoroshiro128,Xorshift1024
from randomgen.dsfmt import DSFMT
from numpy.random import seed
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Reshape,LSTM

seed(1)


class Cell:

    batch = 30
    x_size = 28
    y_size = 28
    in_shape = (x_size,y_size,1)
    epochs = 100


    # in hindsight this was completely useless lol
    # whenever I'd use 'self.x', it didn't work
    # will check that later
    def __init__(self):
        self.batch = batch
        self.x_size = x_size
        self.y_size = y_size
        self.in_shape = in_shape
        self.epochs = epochs

    def trainInit():
        # I thought about it, and... it probably makes more sense
        # to just run the roll internally as opposed to making the
        # roll and seed externally, and then doing an input (WHEN
        # WE REALLY DON'T NEED ONE)...
        
        roll = ran.randint(0,7)
        seed = ran.randint(0,0b1111111111111)

        # we got a switch statement here
        rng = {
            0: ran.seed,
            1: MT19937,
            2: DSFMT,
            3: PCG64,
            4: Philox,
            5: ThreeFry,
            6: Xoroshiro128,
            7: Xorshift1024,
            }
        
        # Handling the fact that Python's rng isn't Randomgen's rng
        if roll == 0:
            ran.seed(seed)
            rg = ran
        else:
            rg = RandomGenerator(rng.get(roll)(seed))

        # turns the output into a dictionary
        out = {
            0: [roll, seed],
            1: rg,
            }
        return out

    def dataGen():
        # this function creates the dataset. stop asking questions
        jhat = []
        for i in range (Cell.batch):
            xhat = []
            d = Cell.trainInit()
            yhat = num.array(d.get(0))
            for j in range (Cell.x_size * Cell.y_size):
                xhat.append(d.get(1).randint(0,255))
            jhat.append(num.array([(num.array(xhat)/256),(yhat/num.array([8,8192]))]))
        jhat = num.array(jhat)
        return jhat

    def dInit():
        # Initializes the dataset (honestly probably could combine this with
        # dataGen() but we're working with hacky code here, gimme a break)

        Cell.dSet = Cell.dataGen()
        Cell.x_train = Cell.dSet[0:,0]
        Cell.y_train = Cell.dSet[0:,1]
        _x = []
        _y = []

        # dataGen sends out wrongly shaped data, so.... here we gooooo
        for i in range(Cell.batch):
            _x = num.append(_x, num.reshape(Cell.x_train[i],[Cell.x_size,Cell.y_size]))
            _y = num.append(_y, num.reshape(Cell.y_train[i],[2]))

        Cell.x_train = num.reshape(_x,[Cell.batch,Cell.x_size,Cell.y_size,1])
        Cell.y_train = num.reshape(_y,[Cell.batch,2])

    def vInit():
        # LITERALLY THE SAME AS DINIT!
        # LITERALLY!!
        # THIS ONLY EXISTS TO CREATE AN **ENTIRELY SEPERATE DATASET**
        # as you can see I'm mad at my incompetetence
        # Honestly I could just generalize this and just run the same function
        # twice but honestly... that just makes too much sense. It isn't at the level
        # of stupid I prefer.
        
        Cell.vSet = Cell.dataGen()
        Cell.x_val = Cell.vSet[0:,0]
        Cell.y_val = Cell.vSet[0:,1]
        _x = []
        _y = []
        for i in range(Cell.batch):
            _x = num.append(_x, num.reshape(Cell.x_val[i],[Cell.x_size,Cell.y_size]))
            _y = num.append(_y, num.reshape(Cell.y_val[i],[2]))
        Cell.x_val = num.reshape(_x,[Cell.batch,Cell.x_size,Cell.y_size,1])
        Cell.y_val = num.reshape(_y,[Cell.batch,2])

    def convInit():
        Cell.model = Sequential()
        Cell.model.add(Conv2D(32, kernel_size=(5,5),
                         strides=(1,1),activation='relu',
                         input_shape=Cell.in_shape,
                              name='C00'))
        Cell.model.add(Conv2D(32, kernel_size=(5,5),
                         strides=(1,1),activation='relu',
                         input_shape=(Cell.x_size,Cell.y_size,1),
                              name='C01'))
        Cell.model.add(Dense(12,activation='relu',input_shape=(12,12,32),
                             name='D1'))
        Cell.model.add(Conv2D(16, kernel_size=(5,5),
                         strides=(1,1),activation='relu',
                         input_shape=(Cell.x_size,Cell.y_size,1),
                              name='C10'))
        Cell.model.add(Conv2D(16, kernel_size=(5,5),
                         strides=(1,1),activation='relu',
                         input_shape=(Cell.x_size,Cell.y_size,1),
                              name='C11'))
        Cell.model.add(Dense(4,activation='relu',input_shape=(4,4,16),
                             name='D2'))
        Cell.model.add(Flatten())
        Cell.model.add(Dense(8,activation='relu',name='D3'))
        Cell.model.add(Dense(2,activation='softmax',name='DOut'))
        Cell.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adam(),
                           metrics=['accuracy'])
    def lstmInit():
        # an alternate model that I was *intending* to make, but uh...
        # sooo much extra work
        Cell.model = Sequential()

    class AccuracyHistory(keras.callbacks.Callback):
        # I just did this because I was told to
        def on_train_begin(self,logs={}):
            self.acc=[]
        def on_epoch_end(self,batch,logs={}):
            self.acc.append(logs.get('acc'))

    def train():
        # trains the model... I guess
        history = Cell.AccuracyHistory()                           
        Cell.model.fit(Cell.x_train, Cell.y_train,
                       batch_size = Cell.batch,
                       epochs = Cell.epochs,
                       verbose=1,
                       validation_data = (Cell.x_val, Cell.y_val),
                       callbacks = [history])
        return Cell.model.evaluate(Cell.x_val,Cell.y_val)
    

D = Cell
D.convInit()

def analTrain(x=1):
    # trains the model a bunch of times, outputs y values and losses
    # for further analysis.
    z = []
    for i in range(x):
        D.dInit()
        D.vInit()
        z.append([D.y_train,D.train()])
    return num.array(z)

