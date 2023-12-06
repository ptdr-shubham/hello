SVM

import numpy as np
import pandas as pd
from sklearn import svm
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

from matplotlib import pyplot
for i in range(9):
  pyplot.subplot(330 + 1 + i)
  pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

train_dataset = train_X.reshape((60000,28*28))

test_dataset =test_X.reshape((10000,28*28))

clf = svm.SVC(decision_function_shape='ovr')

clf.fit(train_dataset, train_y)

 dec = clf.decision_function([train_dataset[1]])
 dec.shape[1]

y_pred=clf.predict(test_dataset)

from sklearn import metrics

print("Accuracy is",metrics.accuracy_score(test_y,y_pred))



class SVM_classifier():

  def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations
    self.lambda_parameter = lambda_parameter
  def fit(self, X, Y):
    self.m, self.n = X.shape
    self.w = np.zeros(self.n)
    self.b = 0
    self.X = X
    self.Y = Y
    for i in range(self.no_of_iterations):
      self.update_weights()
  def update_weights(self):
    y_label = np.where(self.Y <= 0, -1, 1)
    for index, x_i in enumerate(self.X):
      condition = y_label[index] * (np.dot(x_i, self.w) - self.b) >= 1
      if (condition == True):
        dw = 2 * self.lambda_parameter * self.w
        db = 0
      else:
        dw = 2 * self.lambda_parameter * self.w - np.dot(x_i, y_label[index])
        db = y_label[index]

      self.w = self.w - self.learning_rate * dw
      self.b = self.b - self.learning_rate * db
  def predict(self, X):
    output = np.dot(X, self.w) - self.b
    predicted_labels = np.sign(output)
    y_hat = np.where(predicted_labels <= -1, 0, 1)
    return y_hat


clf2=SVM_classifier(0.01,1000,2)

clf2.fit(train_dataset, train_y)

y_pred2=clf2.predict(test_dataset)

print("Accuracy is",metrics.accuracy_score(test_y,y_pred2))




PCA on Gray level image

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('cat.100.jpg',cv2.IMREAD_GRAYSCALE)
image.shape

row_mean = np.mean(image, axis=1)
new_rows = image - row_mean[:, np.newaxis]
cov_row = np.cov(new_rows)
eigenvalues_row, eigenvectors_row = np.linalg.eig(cov_row)
PC_scores_row = np.dot(new_rows.T, eigenvectors_row)

col_mean = np.mean(image,axis=0)
new_cols= image - col_mean
cov_col = np.cov(new_cols)
eigenvalues_col,eigenvectors_col =np.linalg.eig(cov_col)
PC_scores_col = np.dot(new_cols.T,eigenvectors_col)

scores = PC_scores_row + PC_scores_col

new_image = np.dot(scores, eigenvectors_row.T) + row_mean[:, np.newaxis].T

new_image = np.clip(new_image, 0, 255).astype(np.uint8)

cv2.imwrite('new_image.jpg', new_image)




Backward propagation:

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
iteration = 10000

np.random.seed(0)
input_layer_weights = np.random.uniform(size=(input_size, hidden_size))
hidden_layer_weights = np.random.uniform(size=(hidden_size, output_size))
input_layer_bias = np.zeros((1, hidden_size))
hidden_layer_bias = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

for i in range(iteration):
    #forward propogation
    hidden_layer_input = np.dot(X, input_layer_weights) + input_layer_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, hidden_layer_weights) + hidden_layer_bias
    output_layer_output = sigmoid(output_layer_input)
    error = Y - output_layer_output
    #error calulation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(hidden_layer_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    #weight and bias updation
    hidden_layer_weights += hidden_layer_output.T.dot(d_output) * learning_rate
    hidden_layer_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    input_layer_weights += X.T.dot(d_hidden_layer) * learning_rate
    input_layer_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate


test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_input, input_layer_weights) + input_layer_bias), hidden_layer_weights) + hidden_layer_bias)
filter_arr= [1 if x>0.5 else 0 for x in predicted_output]
print(filter_arr)

print(predicted_output)



MP neuron

#AND

w1=1
w2=1

def and_gate(a,b):
  threshold=2;
  if(a*w1+b*w2>=threshold):
    return 1
  else:
    return 0

print(and_gate(0,0))
print(and_gate(0,1))
print(and_gate(1,0))
print(and_gate(1,1))


OR gate

def or_gate(a,b):
  threshold=1;
  if(a*w1+b*w2>=threshold):
    return 1
  else:
    return 0

print(or_gate(0,0))
print(or_gate(0,1))
print(or_gate(1,0))
print(or_gate(1,1))


NOT Gate

def not_gate(a):
  threshold=1;
  if(a*w1>=threshold):
    return 0
  else:
    return 1

print(not_gate(0))
print(not_gate(1))



Perceptron

import numpy as np

def step_function(x):
    return 1 if x >= 0 else 0

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return step_function(summation)

    def train(self, inputs, target_output, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            for input_data, target in zip(inputs, target_output):
                prediction = self.predict(input_data)
                error = target - prediction
                total_error += abs(error)
                self.weights += learning_rate * error * input_data
                self.bias += learning_rate * error
            if total_error == 0:
                break


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_target_output = np.array([0, 0, 0, 1])
or_target_output = np.array([0, 1, 1, 1])
xor_target_output = np.array([0, 1, 1, 0])

or_gate = Perceptron(input_size=2)
or_gate.train(inputs, or_target_output)

and_gate= Perceptron(input_size=2)
and_gate.train(inputs, and_target_output)

xor_gate= Perceptron(input_size=2)
xor_gate.train(inputs, xor_target_output)

print("Testing Perceptron OR gate:")
for input_data in inputs:
    output = or_gate.predict(input_data)
    print(f"Input: {input_data}, Output: {output}")


print("Testing Perceptron AND gate:")
for input_data in inputs:
    output = and_gate.predict(input_data)
    print(f"Input: {input_data}, Output: {output}")


print("Testing Perceptron XOR gate:")
for input_data in inputs:
    output = xor_gate.predict(input_data)
    print(f"Input: {input_data}, Output: {output}")




1. EXPECTATION MAXIMIZATION-

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import scipy.stats.kde as kde

# Generate a dataset with two Gaussian components
mu1, sigma1 = 2, 1
mu2, sigma2 = -1, 0.8
X1 = np.random.normal(mu1, sigma1, size=200)
X2 = np.random.normal(mu2, sigma2, size=600)
X = np.concatenate([X1, X2])
# Plot the density estimation using seaborn
sns.kdeplot(X)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.show()

# Initialize parameters
mu1_hat, sigma1_hat = np.mean(X1), np.std(X1)
mu2_hat, sigma2_hat = np.mean(X2), np.std(X2)
pi1_hat, pi2_hat = len(X1) / len(X), len(X2) / len(X)

# Perform EM algorithm for 20 epochs
num_epochs = 20
log_likelihoods = []
for epoch in range(num_epochs):
	# E-step: Compute responsibilities
	gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
	gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
	total = gamma1 + gamma2
	gamma1 /= total
	gamma2 /= total
	# M-step: Update parameters
	mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
	mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
	sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat)**2) / np.sum(gamma1))
	sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat)**2) / np.sum(gamma2))
	pi1_hat = np.mean(gamma1)
	pi2_hat = np.mean(gamma2)
	# Compute log-likelihood
	log_likelihood = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
								+ pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))
	log_likelihoods.append(log_likelihood)
# Plot log-likelihood values over epochs
plt.plot(range(1, num_epochs+1), log_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Epoch')
plt.show()

# Plot the final estimated density
X_sorted = np.sort(X)
density_estimation = pi1_hat*norm.pdf(X_sorted,mu1_hat,sigma1_hat) + pi2_hat * norm.pdf(X_sorted,mu2_hat,sigma2_hat)
plt.plot(X_sorted, kde.gaussian_kde(X_sorted)(X_sorted), color='green', linewidth=2)
plt.plot(X_sorted, density_estimation, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.legend(['Kernel Density Estimation','Mixture Density'])
plt.show()


2. Image segmentation using k-mean clustering

import cv2
import numpy as np
from math import sqrt
from collections import defaultdict
import matplotlib.pyplot as plt
image = cv2.imread("n.jfif")
print(image.shape)

# pixels  for clustering
data_point=[]
height, width, channels = image.shape
for i in range(height):
  for j in range(width):
    data_point.append(image[i][j])
print("Length of the data" ,len(data_point))
data= np.array(data_point)
print(data[0])

## k -mean
k = 7
max_iteration = 10000
cluster_centroids = [data_point[np.random.randint(len(data_point))] for i in range(k)]
clusters = defaultdict(list)
print(cluster_centroids)
for i in range(max_iteration):
    clusters.clear()
    for r, g, b in data:
        min_distance = 100000
        nearest_centroid = (0, 0, 0)
        for i, j, k in cluster_centroids:
            dist = sqrt((r - i) ** 2 + (g - j) ** 2 + (b - k) ** 2)
            if min_distance > dist:
                min_distance = dist
                nearest_centroid = (i, j, k)
        clusters[nearest_centroid].append((r, g, b))
    new_cluster_points = [np.mean(np.array(point), axis=0) for point in clusters.values()]
    if all(np.array_equal(centroid1, centroid2) for centroid1, centroid2 in zip(cluster_centroids, new_cluster_points)):
        break
    else:
        cluster_centroids = new_cluster_points
print(cluster_centroids)

clusters.clear()
for r, g, b in data:
    min_distance = 100000
    nearest_centroid = (0, 0, 0)
    for i, j, k in cluster_centroids:
        dist = sqrt((r - i) ** 2 + (g - j) ** 2 + (b - k) ** 2)
        if min_distance > dist:
            min_distance = dist
            nearest_centroid = (i, j, k)
    clusters[nearest_centroid].append((r, g, b))

print(clusters.keys())

cluster_centroids = np.uint8(cluster_centroids)

segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(height):
    for j in range(width):
        r, g, b = data_point[i * width + j]
        nearest_centroid_index = np.argmin([np.linalg.norm(np.array([r, g, b]) - centroid) for centroid in cluster_centroids])
        segmented_image[i][j] = cluster_centroids[nearest_centroid_index]

cv2.imwrite("image.jpg", segmented_image)


3. For opening image use pillow or opencv
Reference images are shared in the shared folder, work on the reference images only.

Q-1 From an image create a vector by flattening it completely. Similarly do for a few images and convert into a list of vectors as a Matrix. The final output should be a numpy Matrix.

import numpy as np
from PIL import Image
vector=[]
paths=['cat.100.jpg','cat.101.jpg','cat.102.jpg','cat.103.jpg','cat.104.jpg']
for path in paths:
  img=Image.open(path)
  flatten_image=np.array(img).flatten()
  flatten_image.resize(2200)
  vector.append(flatten_image)
v=np.vstack(vector)
print(v)

Q-2 Compare two images using their histograms. Create histogram for two images in all three channels red, blue, green. Compare the images using histograms across each channel and find if they are equal. Don't use library functions.

import numpy as np
from PIL import Image
def hist(img):
  histogram=np.zeros((256,3),dtype=int)
  for row in img:
    for pix in row:
      r, g ,b=pix
      histogram[r][0]+=1
      histogram[g][1]+=1
      histogram[b][2]+=1
  return histogram
def compare(hist1,hist2):
  s=np.zeros(3,dtype=float)
  for i in range(256):
    s[0]+=min(hist1[i][0],hist2[i][0])
    s[1]+=min(hist1[i][1],hist2[i][1])
    s[2]+=min(hist1[i][2],hist2[i][2])
  total=np.sum(hist1)
  s/=total
  return s
img=Image.open('cat.100.jpg')
img2=Image.open('cat.101.jpg')
h1=hist(np.array(img))
h2=hist(np.array(img2))
print(compare(h1,h2))

Q-3 Take a reference image and apply a sobel filter on the image. Detect the edges and display the output. Don't use library functions for filters.

from PIL import Image
import numpy as np
from scipy.signal import convolve2d
input_image_path = 'cat.100.jpg'
output_image_path = 'output_image.jpg'
image = Image.open(input_image_path)
gray_image = image.convert("L")
image_array = np.array(gray_image)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
filtered_x = convolve2d(image_array, sobel_x, mode='same', boundary='wrap')
filtered_y = convolve2d(image_array, sobel_y, mode='same', boundary='wrap')
edge_image = np.sqrt(filtered_x**2 + filtered_y**2)
edge_image = (edge_image / np.max(edge_image)) * 255
edge_image_pil = Image.fromarray(edge_image.astype(np.uint8))
edge_image_pil.save(output_image_path)
print("Edge-detected image saved!")

Q-4 Write algorithm from scratch without library for histogram of oriented gradients. Apply your algorithm for a reference image and show the resulting feature descriptors.

import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread("cat.100.jpg", cv2.IMREAD_GRAYSCALE)
cell = 8
block =2
#gradiemts
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gradx = cv2.filter2D(img, cv2.CV_64F, sobelx)
grady = cv2.filter2D(img, cv2.CV_64F, sobely)
#magnitude
mag = np.sqrt(gradx**2 + grady**2)
#dir
angle = np.arctan2(grady, gradx) * (180 / np.pi)
angle[angle < 0] += 180
print(mag.shape, angle.shape)
cell_x = mag.shape[0]
cell_y = mag.shape[1]
def create_hist(m,a):
    hist = np.zeros(9)
    bin_width = 20
    for i in range(m.shape[0]):
        for j in range(a.shape[1]):
            new_a = a[i, j]
            weight = (new_a % bin_width) / bin_width
            bin_idx = int(new_a // bin_width) % 9
            hist[bin_idx] += (1 - weight) * m[i, j]
            hist[(bin_idx + 1) % 9] += weight * m[i, j]
    return hist

hists = np.zeros((cell_x,cell_y,9))

for i in range(cell_x):
  for j in range(cell_y):
    cell_mag = mag[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
    cell_angle = angle[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
    hists[i, j, :] = create_hist(cell_mag, cell_angle)
#print(hists)

hog_features = []

for i in range(cell_x - block + 1):
    for j in range(cell_y - block + 1):
        block_histograms = hists[i:i+block, j:j+block, :]
        normalized_block = block_histograms / np.sqrt(np.sum(block_histograms ** 2) + 1e-6)
        hog_features.append(normalized_block.flatten())

#print(np.array(hog_features))

hog = np.array(hog_features)

plt.tight_layout()
fig = plt.figure(figsize=(8,8))
fig.add_subplot(1,2,1)
plt.imshow(img, cmap = "gray")
plt.axis('off')
plt.title("original image")
orient = angle*np.pi/180.0
orient[mag<50]=np.nan
fig.add_subplot(1,2,2)
plt.imshow(orient, cmap = "hsv")
plt.axis('off')
plt.show()

orient.shape

Q-1 Create two random integer matrices of dimension 5*6. Let them be X, Y. Display both matrices. Evaluate the following expression 12*X.XT + 15*(Y.YT)^2 + 108. Display the trace of the resulting matrix. Use numpy.

import numpy as np

X=np.random.randint(10,size=(5,6))
Y=np.random.randint(10,size=(5,6))

print("Matrix X:\n",X)
print("Matrix Y:\n",Y)

res_matrix=12*(np.matmul(X,np.transpose(X)))+15*((np.matmul(Y,np.transpose(Y)))**2)+108
print("Res Matrix:\n",res_matrix)

trace=np.trace(res_matrix)
print("Trace of the matrix is",trace)

Q-2  Take Titanic kaggle CSV dataset. Find the number of children of age below 10 survived along with the total number of children of age below 10. Find the survival percentage of passengers per their passenger class they traveled. What is the percentage of survival between men and women! Use pandas.
import pandas as pd

titanic_data=pd.read_csv("titanic.csv")

children_survived=titanic_data[(titanic_data['Age']<10) & (titanic_data['Survived']==1)].shape[0]
print(children_survived)

total_children=titanic_data[(titanic_data['Age']<10)].shape[0]
print(total_children)

survival=titanic_data.groupby('Pclass')['Survived'].mean()*100
print(survival)

survival2=titanic_data.groupby('Sex')['Survived'].mean()*100
print(survival2)


Q-3 Solve the following nonlinear equations
x^2 + y^3 + 2xy+ 5 = 0
x+y-6 = 0
Use scipy.

import numpy as np
from scipy.optimize import fsolve

def equations(var):
  x,y=var
  eq1=x**2+y**3+2*x*y+5
  eq2=x+y-6
  return [eq1,eq2]

guess=[9,0]

sol=fsolve(equations,guess)

x0,y0=sol

print("Solution Value of x",x0)
print("Solution Value of y",y0)



TIC TAC TOE

import tensorflow as tf
import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.players = ['X', 'O']
        self.current_player = None
        self.winner = None
        self.game_over = False
    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = None
        self.winner = None
        self.game_over = False
    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves
    def make_move(self, move):
        if self.board[move[0]][move[1]] != 0:
            return False
        self.board[move[0]][move[1]] = self.players.index(self.current_player) + 1
        self.check_winner()
        self.switch_player()
        return True
    def switch_player(self):
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]
    def check_winner(self):
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.winner = self.players[int(self.board[i][0] - 1)]
                self.game_over = True
        # Check columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                self.winner = self.players[int(self.board[0][j] - 1)]
                self.game_over = True
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.winner = self.players[int(self.board[0][0] - 1)]
            self.game_over = True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.winner = self.players[int(self.board[0][2] - 1)]
            self.game_over = True

    def print_board(self):
        print("-------------")
        for i in range(3):
            print("|", end=' ')
            for j in range(3):
                print(self.players[int(self.board[i][j] - 1)] if self.board[i][j] != 0 else " ", end=' | ')
            print()
            print("-------------")



game = TicTacToe()
game.current_player = game.players[0]
game.print_board()

while not game.game_over:
    move = input(f"{game.current_player}'s turn. Enter row and column (e.g. 0 0): ")
    move = tuple(map(int, move.split()))
    while move not in game.available_moves():
        move = input("Invalid move. Try again: ")
        move = tuple(map(int, move.split()))
    game.make_move(move)
    game.print_board()

if game.winner:
    print(f"{game.winner} wins!")
else:
    print("It's a tie!")





Q-1  Create Checkers game using Python without utilizing direct game libraries.

from copy import deepcopy
import time
import math

ansi_black = "\u001b[30m"
ansi_red = "\u001b[31m"
ansi_green = "\u001b[32m"
ansi_yellow = "\u001b[33m"
ansi_blue = "\u001b[34m"
ansi_magenta = "\u001b[35m"
ansi_cyan = "\u001b[36m"
ansi_white = "\u001b[37m"
ansi_reset = "\u001b[0m"


class Node:
    def __init__(self, board, move=None, parent=None, value=None):
        self.board = board
        self.value = value
        self.move = move
        self.parent = parent

    def get_children(self, minimizing_player, mandatory_jumping):
        current_state = deepcopy(self.board)
        available_moves = []
        children_states = []
        big_letter = ""
        queen_row = 0
        if minimizing_player is True:
            available_moves = Checkers.find_available_moves(current_state, mandatory_jumping)
            big_letter = "C"
            queen_row = 7
        else:
            available_moves = Checkers.find_player_available_moves(current_state, mandatory_jumping)
            big_letter = "B"
            queen_row = 0
        for i in range(len(available_moves)):
            old_i = available_moves[i][0]
            old_j = available_moves[i][1]
            new_i = available_moves[i][2]
            new_j = available_moves[i][3]
            state = deepcopy(current_state)
            Checkers.make_a_move(state, old_i, old_j, new_i, new_j, big_letter, queen_row)
            children_states.append(Node(state, [old_i, old_j, new_i, new_j]))
        return children_states

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_board(self):
        return self.board

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent


class Checkers:

    def __init__(self):
        self.matrix = [[], [], [], [], [], [], [], []]
        self.player_turn = True
        self.computer_pieces = 12
        self.player_pieces = 12
        self.available_moves = []
        self.mandatory_jumping = False

        for row in self.matrix:
            for i in range(8):
                row.append("---")
        self.position_computer()
        self.position_player()

    def position_computer(self):
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.matrix[i][j] = ("c" + str(i) + str(j))

    def position_player(self):
        for i in range(5, 8, 1):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.matrix[i][j] = ("b" + str(i) + str(j))

    def print_matrix(self):
        i = 0
        print()
        for row in self.matrix:
            print(i, end="  |")
            i += 1
            for elem in row:
                print(elem, end=" ")
            print()
        print()
        for j in range(8):
            if j == 0:
                j = "     0"
            print(j, end="   ")
        print("\n")

    def get_player_input(self):
        available_moves = Checkers.find_player_available_moves(self.matrix, self.mandatory_jumping)
        if len(available_moves) == 0:
            if self.computer_pieces > self.player_pieces:
                print(
                    ansi_red + "You have no moves left, and you have fewer pieces than the computer.YOU LOSE!" + ansi_reset)
                exit()
            else:
                print(ansi_yellow + "You have no available moves.\nGAME ENDED!" + ansi_reset)
                exit()
        self.player_pieces = 0
        self.computer_pieces = 0
        while True:

            coord1 = input("Which piece[i,j]: ")
            if coord1 == "":
                print(ansi_cyan + "Game ended!" + ansi_reset)
                exit()
            elif coord1 == "s":
                print(ansi_cyan + "You surrendered.\nCoward." + ansi_reset)
                exit()
            coord2 = input("Where to[i,j]:")
            if coord2 == "":
                print(ansi_cyan + "Game ended!" + ansi_reset)
                exit()
            elif coord2 == "s":
                print(ansi_cyan + "You surrendered.\nCoward." + ansi_reset)
                exit()
            old = coord1.split(",")
            new = coord2.split(",")

            if len(old) != 2 or len(new) != 2:
                print(ansi_red + "Illegal input" + ansi_reset)
            else:
                old_i = old[0]
                old_j = old[1]
                new_i = new[0]
                new_j = new[1]
                if not old_i.isdigit() or not old_j.isdigit() or not new_i.isdigit() or not new_j.isdigit():
                    print(ansi_red + "Illegal input" + ansi_reset)
                else:
                    move = [int(old_i), int(old_j), int(new_i), int(new_j)]
                    if move not in available_moves:
                        print(ansi_red + "Illegal move!" + ansi_reset)
                    else:
                        Checkers.make_a_move(self.matrix, int(old_i), int(old_j), int(new_i), int(new_j), "B", 0)
                        for m in range(8):
                            for n in range(8):
                                if self.matrix[m][n][0] == "c" or self.matrix[m][n][0] == "C":
                                    self.computer_pieces += 1
                                elif self.matrix[m][n][0] == "b" or self.matrix[m][n][0] == "B":
                                    self.player_pieces += 1
                        break

    @staticmethod
    def find_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n][0] == "c":
                    if Checkers.check_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
                elif board[m][n][0] == "C":
                    if Checkers.check_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if Checkers.check_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

    @staticmethod
    def check_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == "---":
            return False
        if board[via_i][via_j][0] == "C" or board[via_i][via_j][0] == "c":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[old_i][old_j][0] == "b" or board[old_i][old_j][0] == "B":
            return False
        return True

    @staticmethod
    def check_moves(board, old_i, old_j, new_i, new_j):

        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j][0] == "b" or board[old_i][old_j][0] == "B":
            return False
        if board[new_i][new_j] == "---":
            return True

    @staticmethod
    def calculate_heuristics(board):
        result = 0
        mine = 0
        opp = 0
        for i in range(8):
            for j in range(8):
                if board[i][j][0] == "c" or board[i][j][0] == "C":
                    mine += 1

                    if board[i][j][0] == "c":
                        result += 5
                    if board[i][j][0] == "C":
                        result += 10
                    if i == 0 or j == 0 or i == 7 or j == 7:
                        result += 7
                    if i + 1 > 7 or j - 1 < 0 or i - 1 < 0 or j + 1 > 7:
                        continue
                    if (board[i + 1][j - 1][0] == "b" or board[i + 1][j - 1][0] == "B") and board[i - 1][
                        j + 1] == "---":
                        result -= 3
                    if (board[i + 1][j + 1][0] == "b" or board[i + 1][j + 1] == "B") and board[i - 1][j - 1] == "---":
                        result -= 3
                    if board[i - 1][j - 1][0] == "B" and board[i + 1][j + 1] == "---":
                        result -= 3

                    if board[i - 1][j + 1][0] == "B" and board[i + 1][j - 1] == "---":
                        result -= 3
                    if i + 2 > 7 or i - 2 < 0:
                        continue
                    if (board[i + 1][j - 1][0] == "B" or board[i + 1][j - 1][0] == "b") and board[i + 2][
                        j - 2] == "---":
                        result += 6
                    if i + 2 > 7 or j + 2 > 7:
                        continue
                    if (board[i + 1][j + 1][0] == "B" or board[i + 1][j + 1][0] == "b") and board[i + 2][
                        j + 2] == "---":
                        result += 6

                elif board[i][j][0] == "b" or board[i][j][0] == "B":
                    opp += 1

        return result + (mine - opp) * 1000

    @staticmethod
    def find_player_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n][0] == "b":
                    if Checkers.check_player_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_player_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                elif board[m][n][0] == "B":
                    if Checkers.check_player_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_player_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if Checkers.check_player_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_player_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_player_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_player_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

    @staticmethod
    def check_player_moves(board, old_i, old_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j][0] == "c" or board[old_i][old_j][0] == "C":
            return False
        if board[new_i][new_j] == "---":
            return True

    @staticmethod
    def check_player_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == "---":
            return False
        if board[via_i][via_j][0] == "B" or board[via_i][via_j][0] == "b":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[old_i][old_j][0] == "c" or board[old_i][old_j][0] == "C":
            return False
        return True

    def evaluate_states(self):
        t1 = time.time()
        current_state = Node(deepcopy(self.matrix))

        first_computer_moves = current_state.get_children(True, self.mandatory_jumping)
        if len(first_computer_moves) == 0:
            if self.player_pieces > self.computer_pieces:
                print(
                    ansi_yellow + "Computer has no available moves left, and you have more pieces left.\nYOU WIN!" + ansi_reset)
                exit()
            else:
                print(ansi_yellow + "Computer has no available moves left.\nGAME ENDED!" + ansi_reset)
                exit()
        dict = {}
        for i in range(len(first_computer_moves)):
            child = first_computer_moves[i]
            value = Checkers.minimax(child.get_board(), 4, -math.inf, math.inf, False, self.mandatory_jumping)
            dict[value] = child
        if len(dict.keys()) == 0:
            print(ansi_green + "Computer has cornered itself.\nYOU WIN!" + ansi_reset)
            exit()
        new_board = dict[max(dict)].get_board()
        move = dict[max(dict)].move
        self.matrix = new_board
        t2 = time.time()
        diff = t2 - t1
        print("Computer has moved (" + str(move[0]) + "," + str(move[1]) + ") to (" + str(move[2]) + "," + str(
            move[3]) + ").")
        print("It took him " + str(diff) + " seconds.")

    @staticmethod
    def minimax(board, depth, alpha, beta, maximizing_player, mandatory_jumping):
        if depth == 0:
            return Checkers.calculate_heuristics(board)
        current_state = Node(deepcopy(board))
        if maximizing_player is True:
            max_eval = -math.inf
            for child in current_state.get_children(True, mandatory_jumping):
                ev = Checkers.minimax(child.get_board(), depth - 1, alpha, beta, False, mandatory_jumping)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha:
                    break
            current_state.set_value(max_eval)
            return max_eval
        else:
            min_eval = math.inf
            for child in current_state.get_children(False, mandatory_jumping):
                ev = Checkers.minimax(child.get_board(), depth - 1, alpha, beta, True, mandatory_jumping)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha:
                    break
            current_state.set_value(min_eval)
            return min_eval

    @staticmethod
    def make_a_move(board, old_i, old_j, new_i, new_j, big_letter, queen_row):
        letter = board[old_i][old_j][0]
        i_difference = old_i - new_i
        j_difference = old_j - new_j
        if i_difference == -2 and j_difference == 2:
            board[old_i + 1][old_j - 1] = "---"

        elif i_difference == 2 and j_difference == 2:
            board[old_i - 1][old_j - 1] = "---"

        elif i_difference == 2 and j_difference == -2:
            board[old_i - 1][old_j + 1] = "---"

        elif i_difference == -2 and j_difference == -2:
            board[old_i + 1][old_j + 1] = "---"

        if new_i == queen_row:
            letter = big_letter
        board[old_i][old_j] = "---"
        board[new_i][new_j] = letter + str(new_i) + str(new_j)

    def play(self):
        print(ansi_cyan + "##### WELCOME TO CHECKERS ####" + ansi_reset)
        print("\nSome basic rules:")
        print("1.You enter the coordinates in the form i,j.")
        print("2.You can quit the game at any time by pressing enter.")
        print("3.You can surrender at any time by pressing 's'.")
        print("Now that you've familiarized yourself with the rules, enjoy!")
        while True:
            answer = input("\nFirst, we need to know, is jumping mandatory?[Y/n]: ")
            if answer == "Y" or answer == "y":
                self.mandatory_jumping = True
                break
            elif answer == "N" or answer == "n":
                self.mandatory_jumping = False
                break
            elif answer == "":
                print(ansi_cyan + "Game ended!" + ansi_reset)
                exit()
            elif answer == "s":
                print(ansi_cyan + "You've surrendered before the game even started.\nPathetic." + ansi_reset)
                exit()
            else:
                print(ansi_red + "Illegal input!" + ansi_reset)
        while True:
            self.print_matrix()
            if self.player_turn is True:
                print(ansi_cyan + "\nPlayer's turn." + ansi_reset)
                self.get_player_input()
            else:
                print(ansi_cyan + "Computer's turn." + ansi_reset)
                print("Thinking...")
                self.evaluate_states()
            if self.player_pieces == 0:
                self.print_matrix()
                print(ansi_red + "You have no pieces left.\nYOU LOSE!" + ansi_reset)
                exit()
            elif self.computer_pieces == 0:
                self.print_matrix()
                print(ansi_green + "Computer has no pieces left.\nYOU WIN!" + ansi_reset)
                exit()
            elif self.computer_pieces - self.player_pieces == 7:
                wish = input("You have 7 pieces fewer than your opponent.Do you want to surrender?")
                if wish == "" or wish == "yes":
                    print(ansi_cyan + "Coward." + ansi_reset)
                    exit()
            self.player_turn = not self.player_turn


if __name__ == '__main__':
    checkers = Checkers()
    checkers.play()



