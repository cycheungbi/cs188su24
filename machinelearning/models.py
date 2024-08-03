from torch import no_grad, stack
from torch.utils.data import DataLoader
from torch.nn import Module


"""
Functions you should use.
Please avoid importing any other torch functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch.nn import Parameter, Linear
from torch import optim, tensor, tensordot, empty, ones
from torch.nn.functional import cross_entropy, relu, mse_loss
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.

        In order for our autograder to detect your weight, initialize it as a 
        pytorch Parameter object as follows:

        Parameter(weight_vector)

        where weight_vector is a pytorch Tensor of dimension 'dimensions'

        
        Hint: You can use ones(dim) to create a tensor of dimension dim.
        """
        super(PerceptronModel, self).__init__()
        
        # Initialize weight as a Parameter with shape (1, dimensions)
        self.w = Parameter(ones(1, dimensions))

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)

        The pytorch function `tensordot` may be helpful here.
        """
        # Compute dot product using matrix multiplication
        return (x @ self.w.t()).squeeze()

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if self.run(x) >= 0 else -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        You can iterate through DataLoader in order to 
        retrieve all the batches you need to train on.

        Each sample in the dataloader is in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.
        """        
        with no_grad():
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
            while True:
                misclassified = False
                for batch in dataloader:
                    x, y = batch['x'], batch['label']
                    prediction = self.get_prediction(x)
                    if prediction != y.item():
                        self.w += x * y.item()
                        misclassified = True
                if not misclassified:
                    break
            



class RegressionModel(Module):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        # Define the layers of the neural network
        self.layer1 = Linear(1, 64)
        self.layer2 = Linear(64, 64)
        self.layer3 = Linear(64, 1)
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Forward pass through the network
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a tensor of size 1 containing the loss
        """
        # Compute the loss
        predictions = self.forward(x)
        return mse_loss(predictions, y)
 
  

    def train(self, dataset):
        """
        Trains the model.

        In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
        batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

        Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
        is the item we need to predict based off of its features.

        Inputs:
            dataset: a PyTorch dataset object containing data to be trained on
            
        """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(1000):  # Adjust number of epochs as needed
            total_loss = 0
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Compute the loss
                loss = self.get_loss(x, y)
                
                # Backpropagate and update weights
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Print average loss every 100 epochs
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
            
            # Early stopping condition
            if total_loss / len(dataloader) < 0.02:
                print(f"Reached target loss. Stopping training.")
                break


            







class DigitClassificationModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        # Define the layers of the neural network
        self.layer1 = Linear(784, 256)
        self.layer2 = Linear(256, 128)
        self.layer3 = Linear(128, 10)
        
        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a tensor with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Forward pass through the network
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)  # No ReLU in the last layer
        return x

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        # Compute the loss
        predictions = self.run(x)
        return cross_entropy(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        for epoch in range(10):  # Train for up to 10 epochs
            total_loss = 0
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Compute the loss
                loss = self.get_loss(x, y)
                
                # Backpropagate and update weights
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Print average loss and validation accuracy every epoch
            avg_loss = total_loss / len(dataloader)
            val_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            # Early stopping condition
            if val_accuracy >= 0.975:  # Slightly higher threshold for safety
                print(f"Reached target accuracy. Stopping training.")
                break


class LanguageIDModel(Module):
    def __init__(self):
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        
        self.hidden_size = 256  # Increased hidden size
        
        # Define layers
        self.initial_layer = Linear(self.num_chars, self.hidden_size)
        self.recurrent_layer_x = Linear(self.num_chars, self.hidden_size)
        self.recurrent_layer_h = Linear(self.hidden_size, self.hidden_size)
        self.output_layer = Linear(self.hidden_size, len(self.languages))
        
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def run(self, xs):
        batch_size = xs[0].shape[0]
        h = relu(self.initial_layer(xs[0]))
        
        for x in xs[1:]:
            z = self.recurrent_layer_x(x) + self.recurrent_layer_h(h)
            h = relu(z)
        
        return self.output_layer(h)

    def get_loss(self, xs, y):
        predictions = self.run(xs)
        return cross_entropy(predictions, y)

    def train(self, dataset):
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(40):  # Increased from 30 to 40
            total_loss = 0
            for batch in dataloader:
                xs, y = batch['x'], batch['label']
                
                xs = movedim(xs, 1, 0)  # Move sequence length dimension to first position
                
                self.optimizer.zero_grad()
                
                loss = self.get_loss(xs, y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            val_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy >= 0.82:  # Slightly higher threshold for safety
                print(f"Reached target accuracy. Stopping training.")
                break


def Convolve(input: tensor, weight: tensor):
    """
    Acts as a convolution layer by applying a 2d convolution with the given inputs and weights.
    DO NOT import any pytorch methods to directly do this, the convolution must be done with only the functions
    already imported.

    There are multiple ways to complete this function. One possible solution would be to use 'tensordot'.
    If you would like to index a tensor, you can do it as such:

    tensor[y:y+height, x:x+width]

    This returns a subtensor who's first element is tensor[y,x] and has height 'height, and width 'width'
    """
    input_height, input_width = input.shape
    weight_height, weight_width = weight.shape
    output_height = input_height - weight_height + 1
    output_width = input_width - weight_width + 1
    
    Output_Tensor = empty((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            Output_Tensor[i, j] = (input[i:i+weight_height, j:j+weight_width] * weight).sum()
    
    return Output_Tensor



class DigitConvolutionalModel(Module):
    """
    A model for handwritten digit classification using the MNIST dataset.

    This class is a convolutational model which has already been trained on MNIST.
    if Convolve() has been correctly implemented, this model should be able to achieve a high accuracy
    on the mnist dataset given the pretrained weights.


    """    

    def __init__(self):
        # Initialize your model parameters here
        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        # Define additional layers
        self.fc1 = Linear(676, 128)  # 676 = 26 * 26 (output size after convolution)
        self.fc2 = Linear(128, output_size)
        
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)


    def run(self, x):
        """
        The convolutional layer is already applied, and the output is flattened for you. You should treat x as
        a regular 1-dimentional datapoint now, similar to the previous questions.
        """
        x = x.reshape(len(x), 28, 28)
        x = stack(list(map(lambda sample: Convolve(sample, self.convolution_weights), x)))
        x = x.flatten(start_dim=1)
        
        # Apply fully connected layers
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x

 

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a tensor with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss tensor
        """
        predictions = self.run(x)
        return cross_entropy(predictions, y)

        

    def train(self, dataset):
        """
        Trains the model.
        """
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(10):  # Adjust number of epochs as needed
            total_loss = 0
            for batch in dataloader:
                x, y = batch['x'], batch['label']
                
                self.optimizer.zero_grad()
                
                loss = self.get_loss(x, y)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            val_accuracy = dataset.get_validation_accuracy()
            print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            if val_accuracy >= 0.80:
                print(f"Reached target accuracy. Stopping training.")
                break
 