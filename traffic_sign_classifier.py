# Python module imports
import os,glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import cv2 
#%matplotlib inline

"""
How to instantiate traffic_classifier: 
    --> tc = traffic_classifier()
How to load data:
    --> tc.prep_data()
How to visualize 43 unique images:
    --> tc.populate_unique_signs()
    --> tc.visualize_signs()
How to visualize histogram of training data
    --> tc.visualize_hist()
How to visualize the learning curve
    --> tc.__init__(learning_curve=True)
    --> tc.visualize_learning()
How to run with different color space
    --> tc.__init__(yuv=True)
How to train the model
    --> tc.training_model()

Information:
    Data:
        train, valid, and test pickle data sets are loaded also images under WebImages folder
        is loaded.
    Color Space:
        traffic_classifier initialized with YUV color space as a default due to the better accuracy
        but other spaces are also supported. 
    Visulization:
        There are three types of visulizations are available: plotting, histogram, and learning curves.
        1- visualize_sign() function provides three different conditions: plotting unique sings
        limitting number of images, and plotting top prediction statistic images. 
        2- visualize_hist() function plots training data histogram
        3- visualize_learning() function plots learning curve between training and validation data
    Model:
        The training model is based on LeNet but depth increased by 4 and added one more fully
        connected layer and ReLU linear model are used.
        
        

"""

class traffic_classifier(object):
    def __init__(self,norm=False,save_model=False,learning_curve=False,test_model=False,new_image=False,
                 rotation=False,epochs=5,learning_rate=0.001,keep_prob=1.0,
                 gray=False,xyz=False,hsv=False,hls=False,lab=False,luv=False,ycrcb=False,yuv=False):
        # Extra capability parameters
        self.norm = norm
        self.save_model = save_model
        self.learning_curve = learning_curve
        self.test_model = test_model
        self.new_image = new_image
        self.rotation = rotation
        # Color Spaces        
        self.gray = gray        
        self.xyz = xyz
        self.hsv = hsv
        self.hls = hls
        self.lab = lab
        self.luv = luv
        self.yuv = yuv
        self.ycrcb = ycrcb        
        # Input Files
        self.training_file = "train.p" #Training data
        self.validation_file = "valid.p" #Validation data
        self.testing_file = "test.p" #Test data once the model is trained
        self.signnames = "signnames.csv" #Traffic Sign Indexes
        self.webimage_dir = r".\WebImages" #Randomly picked images in .jpg format
        # Loading data into list
        self.X_train = [] # used for image conversion
        self.X_train_pickle = [] # training data 
        self.y_train_pickle = [] # label training data 
        self.X_valid_pickle = [] # validation data 
        self.y_valid_pickle = [] # validation label data 
        self.X_test = [] # test data 
        self.y_test = [] # test label data 
        self.X_web_dir = [] # location of random jpg images       
        self.X_web = []  # Ramdom image test set      
        self.y_web = []  # Random image test label 
        self.n_classes = len(pd.read_csv(self.signnames)) # Number of classes
        self.sign_csv = [] # Reading csv traffic sign file into a list
        self.sign_index = [] # Sign indexes
        self.learning = [] # Training and Validation learning
        self.prep_data() # loading data
        # Create dictionary for unique traffic signals
        self.dict_unique_signs = {} # each classID is presented: total 43 classes
        # Visualization parameters how many rows and columns of subplots
        self.plot_row = 8 # row numbers
        self.plot_col = 6 # column numbers
        self.title = "German Traffic Signal Distribution" # title of histogram
        # Model and Achitecture Information
        # --> Convolution Layer1 
        if (self.gray): # Gray color space only 1-channel 
            self.conv1_shape = (5, 5, 1, 24) # Filter shape gray color space
        else:
            self.conv1_shape = (5, 5, 3, 24) # Filter shape for color space
        self.conv1_strides=[1, 1, 1, 1] # Convolution Layer1 Strides
        # --> Pooling Layer1
        self.conv1_pooling_ksize=[1, 2, 2, 1] # Kernel size
        self.conv1_pooling_strides=[1, 2, 2, 1] # Pooling strides
        # --> Convolution Layer2
        self.conv2_shape = (5, 5, 24, 64) # Filter shape for color space
        self.conv2_strides=[1, 1, 1, 1] #Convolution Layer2 Strides
        # --> Pooling Layer2
        self.conv2_pooling_ksize=[1, 2, 2, 1] # Kernel size
        self.conv2_pooling_strides=[1, 2, 2, 1] # Pooling strides
        # --> Dropout
        self.keep_prob=keep_prob # Dropping 50% of the data
        # --> Fully connected Layer1
        self.fc1_shape = (1600, 480) # flattening the connections
        # --> Fully connected Layer2
        self.fc2_shape = (480, 168) # flattening the connections
        # --> Fully connected Layer3
        self.fc3_shape = (168,84)   # flattening the connections     
        # --> Fully connected Layer4
        self.fc4_shape = (84, self.n_classes) # flattening the connections         
        # Model training variables        
        self.mu = 0 
        self.sigma = 0.1
        self.rate = learning_rate
        self.EPOCHS = epochs
        if (self.EPOCHS <= 10):
            self.epoch_print = 1;
        elif (self.EPOCHS <= 50):
            self.epoch_print = 10
        else:
            self.epoch_print = 50
        self.BATCH_SIZE = 128        
        # Training Pipeline Variables
        if (self.gray): # Gray color space has only 1 channel
            self.X = tf.placeholder(tf.float32, (None, self.X_test.shape[1], self.X_test.shape[2], 1)) # Creating place holder for X
        else: # Color Space has 3 channels
            self.X = tf.placeholder(tf.float32, (None, self.X_test.shape[1], self.X_test.shape[2], 3)) # Creating placeholder for X
        self.y = tf.placeholder(tf.int32, (None)) # Creating placeholder for y
        self.one_hot_y = tf.one_hot(self.y, self.n_classes)
        # Random image parameters
        self.top_pred = [] # top predictions
        self.k = 5 # number of top predictions
        
    # Function for loading data
    # self.X_train_pickle --> training features
    # self.y_train_pickle --> training labels
    # self.X_valid_pickle --> validation features
    # self.y_valid_pickle --> validation labels
    # self.X_test --> test features
    # self.y_test --> test labels
    # self.X_web --> random images
    # self.y_web --> random labels
    def prep_data(self):
        # Pickle data loading
        self.sign_csv = pd.read_csv('signnames.csv')
        self.sign_index = self.sign_csv.iloc[:,1]
        with open(self.training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(self.validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(self.testing_file, mode='rb') as f:
            test = pickle.load(f)
        self.X_train_pickle, self.y_train_pickle = train['features'], train['labels']
        self.X_valid_pickle, self.y_valid_pickle = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']
        # Random image data loading
        for web_image in glob.glob(os.path.join(self.webimage_dir,r'*.jpg')):
            web_sign_labels = web_image.split('\\')[-1]            
            web_sign_labels = web_sign_labels.split('.')[0]            
            self.y_web.append(int(web_sign_labels.split('_')[-1]))  
            image = cv2.imread(web_image)
            self.X_web.append(image)
            self.X_web_dir.append(web_image)
    # Function for data summary       
    # Printing number of samples for training, validation, and test data set
    # Printing shape of images example (32x32x3)
    # Printing unique labels
    def data_summary(self):
        print("Number of training samples = {}".format(self.X_train_pickle.shape[0]))
        print("Number of validation samples = {}".format(self.X_valid_pickle.shape[0]))
        print("Number of test samples = {}".format(self.X_test.shape[0]))
        print ("Number of random images for testing = {}".format(np.array(self.X_web).shape[0]))
        print ("Image shape for training, validation, and testing samples = {}".format(self.X_train_pickle.shape[1:4]))
        print("Number of classes =", self.n_classes)
    # Function to create a dictionary for unique traffic signs
    def populate_unique_signs(self):
        for i in range(self.X_train_pickle.shape[0]):
            if (len(self.dict_unique_signs) == len(np.unique(self.y_train_pickle)) ):
                return
            else:
                self.dict_unique_signs[self.y_train_pickle[i]] = self.X_train_pickle[i]    
    # Function to get a slice from image data
    # X --> Image
    # y --> Signal Definition
    # num_samples --> the number of the samples
    def get_samples(self,X,y, num_samples):
        X,y = shuffle(X,y)
        return X[:num_samples],y[:num_samples]
    # Helper function to plot unique traffic signs and also plot image data
    # cols --> the number of collumn plots
    # X --> image data
    # y --> image labels
    # img_dict --> prepopulated unique image dictinary visulization
    # top_pred --> top prediction number for traffic sign class id prediction
    def visualize_signs(self,cols=0,X=None,y=None,img_dict=False,top_pred=None):
        if (img_dict and X==None):        
            plt.figure(figsize=(2.5*self.plot_col,2.5*self.plot_row))        
            for i in range(len(self.dict_unique_signs)):
                plt.subplot(self.plot_row,self.plot_col,i+1)
                plt.imshow(self.dict_unique_signs[i])
                plt.text(0, 0, '{}: {}'.format(i, self.sign_index[i]), color='k'
                ,backgroundcolor='gray', fontsize=9)   
                plt.text(0, self.dict_unique_signs[i].shape[0]-3, '{}'.format(self.dict_unique_signs[i].shape), color='k'
                ,backgroundcolor='g', fontsize=9) 
        elif (X != None):
            rows = len(X)/cols
            plt.figure(figsize=(2.5*cols,2.5*rows))
            for i in range(len(X)):
                if ((len(X)%cols) > 0):
                    plt.subplot(rows+1,cols,i+1)
                else:
                    plt.subplot(rows,cols,i+1)
                    plt.imshow(X[i][:,:,0],cmap='gray')
                plt.text(0, 0, '{}:{}'.format(y[i],self.sign_index[y[i]]), color='k'
                ,backgroundcolor='c', fontsize=8)   
                plt.text(0, np.array(X[i]).shape[0]-3, '{}'.format(np.array(X[i]).shape), color='k'
                ,backgroundcolor='y', fontsize=9) 
        elif (top_pred != None):
            rows = int(len(self.top_pred.indices)/cols)
            plt.figure(figsize=(3.5*cols,3.5*rows))
            for i in range(rows):
                if ((rows%cols) > 0):
                    plt.subplot(rows+1,cols,i+1)
                else:
                    plt.subplot(rows,cols,i+1)
                for j in range(self.k):
                    if (self.y_web[j] != top_pred.indices[j][0]):
                        plt.title("Wrong Prediction: {} ".format(self.sign_index[top_pred.indices[i][j]], color='red'))
                        plt.text(np.array(X[i]).shape[0]+3,5*(j+1),"Prediction: ({:.2f}%) --> {}".format(100*top_pred.values[i][j],self.sign_index[top_pred.indices[i][j]]))
                    else:
                        plt.title("Right Prediction: {} ".format(self.sign_index[top_pred.indices[i][j]],color='blue'))
                        plt.text(np.array(X[i]).shape[0]+3,5*(j+1),"Prediction: ({:.2f}%) --> {}".format(100*top_pred.values[i][j],self.sign_index[top_pred.indices[i][j]]))
                        
                plt.imshow(X[i])
        plt.show()
    # Visulization of histogram of training data
    def visualize_hist(self):
        plt.figure(figsize=(15, 5))
        plt.title('{}'.format(self.title))
        plt.ylabel('{}'.format("Frequency"))
        plt.xlabel('{}'.format("Traffic Sign Labels"))
        plt.hist(self.y_train_pickle,bins=self.n_classes,facecolor='blue')
        plt.grid(True)        
        plt.show()
    # Visualizing learning curve for training and validation data
    def visualize_learning(self,learning):
        np_learning = np.array(learning)
        epochs = np_learning[:,0]
        train = np_learning[:,1]
        valid = np_learning[:,2]
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, train, label='training')
        plt.plot(epochs, valid, label='validation')
        plt.title('Learning')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.xticks(np.arange(0,self.EPOCHS,self.epoch_print))
        plt.legend(loc='upper left')
    # Reading image
    def read_image(self,file):
        return plt.imread(file)
    # Resizing by 32x32x3
    def resize_image(self,image):
        return cv2.resize(image,self.X_test.shape[1:3])
    # Loading images 
    def load_image(self):
        prep_image = lambda file: self.resize_image(self.read_image(file))
        [self.X_train.append(prep_image(self.X[i]) ) for i in range(self.X.shape[0])]        
    # Normalize images
    def image_normalization(self,X):
        return (X - X.mean())/X.std()
    # Rotating images
    def image_random_rotation(self,image, angle):
        if angle == 0:
            return image
        angle = np.random.uniform(-angle, angle)
        rows, cols = image.shape[:2]
        size = np.array(image).shape[:2]
        center = cols/2, rows/2
        scale = 1.0
        rotation = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(image, rotation, size)
    # Image augmentation
    def augment_image(self,image,angle):
        if (self.norm):
            image = self.image_normalization(image)
        if(self.rotation):
            image = self.image_random_rotation(image, angle)
            pass
        if (self.gray):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        elif(self.xyz): 
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2XYZ)
        elif (self.hsv):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
        elif (self.hls):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2HSV)
        elif (self.lab):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2Lab)
        elif (self.luv):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2Luv)
        elif (self.ycrcb):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YCrCb)  
        elif (self.yuv):
            image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2YUV) 
        return image
    # Modifyin image color space and rotation   
    def modify_image(self,X,test_data=False):
        tmp = []
        augmenter = lambda x: self.augment_image(x, angle=15,)
        [tmp.append(augmenter(X[i])) for i in range(X.shape[0])]
        if (self.gray):
            print ("Color: Gray \n")
            tmp = np.reshape(tmp,(-1,32,32,1))
        elif (self.xyz):
            print ("Color: XYZ \n")
            tmp = np.reshape(tmp,(-1,32,32,3))
        elif (self.hsv):
            print ("Color: HSV \n")
            tmp = np.reshape(tmp,(-1,32,32,3))
        elif (self.hls):
            print ("Color: HLS Image \n")
            tmp = np.reshape(tmp,(-1,32,32,3))        
        elif (self.lab):
            print ("Color: LAB \n")
            tmp = np.reshape(tmp,(-1,32,32,3))   
        elif (self.luv):
            print ("Color: LUV \n")
            tmp = np.reshape(tmp,(-1,32,32,3))   
        elif (self.ycrcb):
            print ("Color: YCrCb \n")
            tmp = np.reshape(tmp,(-1,32,32,3))          
        elif (self.yuv):
            print ("Color: YUV \n")
            tmp = np.reshape(tmp,(-1,32,32,3))         
        return tmp
     # Model Specifications    
    def LeNet(self,x):    
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        
        # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x24.
        conv1_W = tf.Variable(tf.truncated_normal(shape=self.conv1_shape, mean = self.mu, stddev = self.sigma))
        conv1_b = tf.Variable(tf.zeros(self.conv1_shape[3]))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=self.conv1_strides, padding='VALID') + conv1_b
    
        # SOLUTION: Activation.
        conv1 = tf.nn.relu(conv1)
    
        # SOLUTION: Pooling. Input = 28x28x24. Output = 14x14x24.
        conv1 = tf.nn.max_pool(conv1, ksize=self.conv1_pooling_ksize, strides=self.conv1_pooling_strides, padding='VALID')
    
        # SOLUTION: Layer 2: Convolutional. Output = 10x10x64.
        conv2_W = tf.Variable(tf.truncated_normal(shape=self.conv2_shape, mean = self.mu, stddev = self.sigma))
        conv2_b = tf.Variable(tf.zeros(self.conv2_shape[3]))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=self.conv2_strides, padding='VALID') + conv2_b
        
        # SOLUTION: Activation.
        conv2 = tf.nn.relu(conv2)
    
        # SOLUTION: Pooling. Input = 10x10x64. Output = 5x5x64.
        conv2 = tf.nn.max_pool(conv2, ksize=self.conv2_pooling_ksize, strides=self.conv2_pooling_strides, padding='VALID')
    
        # Dropout
        conv2 = tf.nn.dropout(conv2, self.keep_prob)        
       
       # SOLUTION: Flatten. Input = 5x5x64. Output = 1600.
        fc0   = flatten(conv2)
        
        # SOLUTION: Layer 3: Fully Connected. Input = 1600. Output = 480.
        fc1_W = tf.Variable(tf.truncated_normal(shape=self.fc1_shape, mean = self.mu, stddev = self.sigma))
        fc1_b = tf.Variable(tf.zeros(self.fc1_shape[1]))
        fc1   = tf.matmul(fc0, fc1_W) + fc1_b
        
        # SOLUTION: Activation.
        fc1    = tf.nn.relu(fc1)
    
        # SOLUTION: Layer 4: Fully Connected. Input = 480. Output = 168.
        fc2_W  = tf.Variable(tf.truncated_normal(shape=self.fc2_shape, mean = self.mu, stddev = self.sigma))
        fc2_b  = tf.Variable(tf.zeros(self.fc2_shape[1]))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        
        # SOLUTION: Activation.
        fc2    = tf.nn.relu(fc2)
    
        # SOLUTION: Layer 5: Fully Connected. Input = 168. Output = 84.
        fc3_W  = tf.Variable(tf.truncated_normal(shape=self.fc3_shape, mean = self.mu, stddev = self.sigma))
        fc3_b  = tf.Variable(tf.zeros(self.fc3_shape[1]))
        fc3 = tf.matmul(fc2, fc3_W) + fc3_b
        
        # SOLUTION: Activation.
        fc3 = tf.nn.relu(fc3)
        
        # SOLUTION: Layer 6: Fully Connected. Input = 84. Output = 43.
        fc4_W  = tf.Variable(tf.truncated_normal(shape=self.fc4_shape, mean = self.mu, stddev = self.sigma))
        fc4_b  = tf.Variable(tf.zeros(self.fc4_shape[1]))
        logits = tf.matmul(fc3, fc4_W) + fc4_b
        
        return logits
    # Training model and testing
    def training_pipeline(self):

        logits = self.LeNet(self.X)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y,logits=logits )
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate = self.rate)
        training_operation = optimizer.minimize(loss_operation)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        y_pred = tf.nn.softmax(logits)
        pred = tf.nn.top_k(y_pred,k=self.k)
        saver = tf.train.Saver()
    
        def evaluate(self,X_data, y_data):
            num_examples = len(X_data)
            total_accuracy = 0
            sess = tf.get_default_session()
            for offset in range(0, num_examples, self.BATCH_SIZE):
                batch_x, batch_y = X_data[offset:offset+self.BATCH_SIZE], y_data[offset:offset+self.BATCH_SIZE]
                accuracy = sess.run(accuracy_operation, feed_dict={self.X: batch_x, self.y: batch_y})
                total_accuracy += (accuracy * len(batch_x))
            return total_accuracy / num_examples

        def evaluate_model_with_new_img(self):
            with tf.Session() as sess:
                saver.restore(sess, tf.train.latest_checkpoint('.'))
                feed_dict = {self.X: self.X_web, self.y: self.y_web}
                self.top_pred = sess.run(pred, feed_dict=feed_dict)  
                #print(top_pred.values)
                #print(top_pred.indices)                      
        def evaluate_model(self):
            with tf.Session() as sess:
                saver.restore(sess, tf.train.latest_checkpoint('.'))
                test_accuracy = evaluate(self,self.X_test, self.y_test)
                print("Test Accuracy = {:.3f}".format(test_accuracy))
      
        def training_model(self):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                num_examples = len(self.X_train_pickle)
                
                print("Training...")
                print()
                for i in range(self.EPOCHS):
                    self.X_train_pickle, self.y_train_pickle = shuffle(self.X_train_pickle, self.y_train_pickle)
                    #self.X_train_pickle, self.y_train_pickle = shuffle(self.X_train_pickle, self.y_train_pickle)                    
                    for offset in range(0, num_examples, self.BATCH_SIZE):
                        end = offset + self.BATCH_SIZE
                        batch_x, batch_y = self.X_train_pickle[offset:end], self.y_train_pickle[offset:end]
                        #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
                        sess.run(training_operation, feed_dict={self.X: batch_x, self.y: batch_y})
                    
                    training_accuracy = evaluate(self,self.X_train_pickle, self.y_train_pickle)    
                    validation_accuracy = evaluate(self,self.X_valid_pickle, self.y_valid_pickle)
                    self.learning.append([i+1,training_accuracy,validation_accuracy])
                    #self.visualize_learning(accuracy=training_accuracy,training=True, epoch=i)
                    if ((i+1)%self.epoch_print == 0):                    
                        print("EPOCH {} ...".format(i+1))
                        print("Training Accuracy = {:.3f}".format(training_accuracy))                        
                        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                        print()
                if (self.save_model):    
                    saver.save(sess, './lenet')
                    print("Model saved")
                if (self.learning_curve):
                    self.visualize_learning(self.learning)
                if (self.test_model):
                    evaluate_model(self)
                if (self.new_image):
                    evaluate_model_with_new_img(self)
            
        return training_model(self)
            


tc = traffic_classifier()
#tc.__init__(save_model=True,test_model=True,new_image=True)
#tc.load_image()
#tc.load_data()
tc.data_summary()
#"""
tc.populate_unique_signs()
#tc.visualize_signs(img_dict=True)
tc.__init__(gray=True)
X,y = tc.get_samples(tc.X_train_pickle,tc.y_train_pickle,5)
tc.visualize_signs(5,X,y)
#tc.visualize_hist()
#"""
#X = tc.modify_image(X)
#tc.visualize_plots(5,X,y)
"""
tc.X_train_pickle = tc.modify_image(tc.X_train_pickle)
tc.X_valid_pickle = tc.modify_image(tc.X_valid_pickle)
tc.X_test = tc.modify_image(tc.X_test,test_data=True)

tc.training_pipeline()
"""