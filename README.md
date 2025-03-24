# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the FashionMNIST dataset. The FashionMNIST dataset contains grayscale images of 10 different clothing categories (e.g., T-shirt, trousers, dress, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while ensuring computational efficiency.
## Neural Network Model

![image](https://github.com/user-attachments/assets/6c9567d9-f2c1-4752-ac3f-747fdd431aec)

## DESIGN STEPS

### STEP 1: Problem Statement
Define the goal of classifying fashion apparel items into 10 categories using a CNN. Ensure high accuracy while maintaining efficiency.

### STEP 2: Dataset Collection
Use the FashionMNIST dataset, which contains 60,000 training and 10,000 test images of clothing items. Each image is 28×28 grayscale and labeled with one of 10 classes.

### STEP 3: Data Preprocessing
Convert images to tensors and normalize pixel values to [0,1]. Use DataLoaders for efficient batch processing during training and testing.

### STEP 4: Model Architecture
Design a CNN with convolutional layers for feature extraction, ReLU activation, pooling layers for downsampling, and fully connected layers for classification.

### STEP 5: Model Training
Train the CNN using CrossEntropyLoss and Adam optimizer for multiple epochs. Monitor accuracy and loss to ensure proper learning.

### STEP 6: Model Evaluation
Test the model on unseen FashionMNIST data, compute performance metrics, and analyze using a confusion matrix.

### STEP 7: Model Deployment & Visualization
Save the trained model for future use and visualize predictions on sample test images. Optionally, integrate into an application for real-world use.

## PROGRAM

### Name:Aravind Kumar SS
### Register Number:212223110004
```python
class CNNClassifier(nn.Module):
  def __init__(self): # Define __init__ method explicitly
    super(CNNClassifier, self).__init__() # Call super().__init__() within __init__
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # Correct argument names
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Correct argument names
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Correct argument names
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjust input size for Linear layer (Calculation needs update if image size changed)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x))) # Correctly call self.conv1
    x = self.pool(torch.relu(self.conv2(x)))  # Correctly call self.conv2
    x = self.pool(torch.relu(self.conv3(x))) # Correctly call self.conv3
    x = x.view(x.size(0), -1) # Flatten the tensor
    x = torch.relu(self.fc1(x)) # Correctly call self.fc1
    x = torch.relu(self.fc2(x)) # Correctly call self.fc2
    x = self.fc3(x)
    return x
```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

```

```python
# Train the Model
## Step 3: Train the Model
def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    print('Name: Aravind Kumar SS')
    print('Register Number: 212223110004')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print only once per epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

```

## OUTPUT
### Training Loss per Epoch

![Screenshot 2025-03-24 092535](https://github.com/user-attachments/assets/c1361995-988a-4718-9a22-895042cacda7)


### Confusion Matrix

![Screenshot 2025-03-24 092554](https://github.com/user-attachments/assets/cbdc3437-350e-4a54-8842-1dfc2cf64217)


### Classification Report

![Screenshot 2025-03-24 092622](https://github.com/user-attachments/assets/f5391a33-f669-4ae8-af57-7626f5c446e2)



### New Sample Data Prediction

![Screenshot 2025-03-24 103735](https://github.com/user-attachments/assets/439bb092-3805-4110-9131-3e8adbd4661b)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
