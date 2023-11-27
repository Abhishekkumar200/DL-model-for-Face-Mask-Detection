# DL model for Lightweight Face Mask Detection
This Project contains DL model optimization for Lightweight Face Mask Detection, written in Python using PyTorch.

## Dataset
The dataset consists of two folder images and annotations. Images folder contains 853 PNG 
images and annotations contains 853 XML files respect to each images in the images folder.

The images belong to the three classes :
* With mask
* Without mask
* Mask worn incorrectly

Images are in png format here, and annotations are in xml. In order to make working with 
XML feel like working with JSON, a Python module called xmltodict can be added. With its 
unparse function, xmltodict also enables repeated trips back to XML.

<img width="344" alt="image" src="https://github.com/Abhishekkumar200/DL-model-for-Face-Mask-Detection/assets/84954320/1ea7234a-dbe4-43c2-91ec-af88e24232cb">
<img width="450" height="235" alt="image" src="https://github.com/Abhishekkumar200/DL-model-for-Face-Mask-Detection/assets/84954320/c0a28e56-07bb-4765-bcc0-777dd06cde7d">

## Data Pre-processing
* First: we will go through the data folder “mask_and_without_mask”, and collects the 
  names of all image and XML files in that directory tree. Using for loop we will iterates 
  over the filenames list to collect the names of the image and XML files in that 
  directory. For each filename in filenames, the code checks if the last three characters 
  of the filename are "xml". If so, it appends the filename to xml_names; otherwise, it 
  appends the filename to img_names. At the end, img_names will contain a list of 
  names of all image files in the directory tree and xml_names will contain a list of 
  names of all XML files in that directory tree.
* Then we reads the XML files associated with the image files in the img_names list, 
  extracts the object names from those XML files, and counts the number of 
  occurrences of each object name.
* We then defines a function dataset_creation() which takes a list of image names as 
  input. It then loops through each image, opens the corresponding XML file to extract 
  the object label and bounding box coordinates. It then crops the image using the 
  bounding box coordinates and applies a series of transformations to the cropped 
  image (resizing, conversion to tensor, and normalization). The transformed image 
  and label are then appended to separate lists. Finally, the function returns a tuple of 
  the two lists as a final dataset. The code also defines a transform object that specifies 
  the sequence of transformations to be applied to the image. Finally, the dataset 
  variable is assigned the output of the dataset_creation() function, using img_names 
  as the input list.
* At the end we splits the given dataset into training and testing sets. The size of the 
  training set is set to 70% of the original dataset size, and the remaining 30% is 
  allocated for the test set. The batch size for the dataloaders is set to 16, and the 
  training and testing datasets are passed to PyTorch's DataLoader function with the 
  appropriate parameters (i.e., batch_size, shuffle). The train_loader and test_loader 
  variables store the resulting dataloaders for the training and testing sets, 
  respectively.

## Images Identification with Target Class

As there are 3 classes in target, we can use 3 colors for cascading the face.

* $${\textcolor{red}{Red}}$$ ➜ without_mask
* $${\textcolor{green}{Green}}$$ ➜ with_mask
* $${\textcolor{yellow}{Yellow}}$$ ➜ mask_weared_incorrect

For visualizing the annotated objects in a dataset of images with associated XML annotation 
files. We defines a function face_cas that takes an image filename as input, reads the 
corresponding XML annotation file, and plots the image with bounding boxes around the 
objects in the image. The bounding boxes are colored according to the class label of the 
object (with_mask, without_mask, or mask_weared_incorrect).

The function first reads the XML file associated with the input image using xmltodict, then 
reads the image using plt.imread. It then creates a plot using plt.subplots, adds the image to 
the plot using ax.imshow, and adds bounding boxes and class labels to the plot using 
mpatches.Rectangle and ax.annotate.

The code then creates a copy of the img_names list called fun_images, and calls the 
face_cas function on the first seven images in fun_images.
<img width="456" alt="image" src="https://github.com/Abhishekkumar200/DL-model-for-Face-Mask-Detection/assets/84954320/5928d24e-333a-4fdc-8977-eb248fcbe966">

## Model
Our model consists of four convolutional layers with batch normalization, ReLU activation, 
and max-pooling. Each convolutional layer has a kernel size of 3x3 and padding of 1. The first 
convolutional layer takes input with 3 channels (RGB images). After the convolutional layers, 
the output is flattened and passed through two fully connected (linear) layers with ReLU 
activation. The output of the second fully connected layer is the final output of the model, 
which predicts the class of the input image (with 3 possible classes).

#### Code:

```python
class MaskDetector(nn.Module):
 def __init__(self):
 super(MaskDetector, self).__init__()
 self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
 self.bn1 = nn.BatchNorm2d(8)
 self.relu1 = nn.ReLU(inplace=True)
 self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
 self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
 self.bn2 = nn.BatchNorm2d(16)
 self.relu2 = nn.ReLU(inplace=True)
 self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
 self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
 self.bn3 = nn.BatchNorm2d(32)
 self.relu3 = nn.ReLU(inplace=True)
 self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
 self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
 self.bn4 = nn.BatchNorm2d(64)
 self.relu4 = nn.ReLU(inplace=True)
 self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
 self.flatten = nn.Flatten()
 self.fc1 = nn.Linear(64 * 14 * 14, 512)
 self.relu5 = nn.ReLU(inplace=True)
 self.fc2 = nn.Linear(512, 3)

 def forward(self, x):
 x = self.conv1(x)
 x = self.bn1(x)
 x = self.relu1(x)
 x = self.pool1(x)
 x = self.conv2(x)
 x = self.bn2(x)
 x = self.relu2(x)
 x = self.pool2(x)
 x = self.conv3(x)
 x = self.bn3(x)
 x = self.relu3(x)
 x = self.pool3(x)
 x = self.conv4(x)
 x = self.bn4(x)
 x = self.relu4(x)
 x = self.pool4(x)
 x = self.flatten(x)
 x = self.fc1(x)
 x = self.relu5(x)
 x = self.fc2(x)
 return x
```

The model consists of four convolutional layers, each followed by a batch normalization 
layer, a ReLU activation function, and a max pooling layer. The convolutional layers are used 
to extract relevant features from the input image, while the batch normalization layers 
normalize the activations and improve the stability and speed of the training process. The 
ReLU activation functions introduce non-linearity to the model, allowing it to learn complex 
representations. The max pooling layers are used to downsample the feature maps, 
reducing the spatial dimensions and preserving the most important features.

After the four convolutional layers, the output is flattened and passed through two fully 
connected layers with ReLU activation functions. The first fully connected layer has 512 
units and the second one has 3 units (one for each class). The fully connected layers are 
used to combine the extracted features and make the final prediction for the input image.

In the forward pass, the input image is passed through each layer in order, with the output 
of each layer being fed as input to the next layer. Finally, the output of the last fully 
connected layer is returned as the predicted probability distribution over the three classes. 
During the training process, the model is optimized using a suitable loss function (such as 
cross-entropy loss) and a suitable optimizer (such as stochastic gradient descent). The 
weights of the model are updated using backpropagation

**The choices made for the different layers and activation functions in this MaskDetector 
model have been made based on the following considerations:**

* In this model, we have used four convolutional layers with increasing number of 
  filters. The first layer has 16 filters, the second has 32 filters, the third has 64 filters 
  and the last layer has 128 filters. The kernel size used in each layer is 3x3, which is a 
  common choice in many image classification tasks. The stride is set to 1 and padding 
  is set to 1, which ensures that the spatial dimensions of the feature maps are 
  preserved throughout the convolutional layers. The increasing number of filters 
  allows the model to learn more complex features as we move through the layers.
* Batch normalization layers are used to normalize the activations of the previous 
  layer, which improves the stability and speed of the training process. By normalizing 
  the activations, batch normalization layers can reduce the effect of the vanishing and 
  exploding gradients problem, which can occur during the training of deep neural 
  networks. In this model, batch normalization layers are added after each 
  convolutional layer.
* we have used the Rectified Linear Unit (ReLU) activation function after each batch 
  normalization layer. ReLU is a popular choice for image classification tasks because it 
  is computationally efficient and has been shown to perform well in practice.
* we have used max pooling layers with a kernel size of 2x2 and a stride of 2. This 
  reduces the spatial dimensions of the feature maps by a factor of 2 at each layer, 
  which allows the model to focus on the most important features.
* The first fully connected layer has 512 units and the second one has 3 units (one for 
  each class). The ReLU activation function is used after the first fully connected layer. 
  Fully connected layers are used to combine the extracted features and make the 
  final prediction for the input image.

## Training the Mode
We trained our model for 50 epochs on the training datasets and we get the following 
results:
`Epoch 50 Loss: 0.0093`
At the 50th epoch the training loss is 0.0093 and training accuracy is 99.6842 %.

## Testing the model
While testing the model on test datasets we get the following results:
* Test Loss: 0.2301 
* Test Accuracy: 95.9902 %
* Best Testing Accuracy: 95.9902 %
