# Machine Learning for Image Scaling
## Machine Learning Engineer Nanodegree
Lucas Dupin  
September 21st, 2016

## I. Definition

### Project Overview
Image resizing is a recurring problem for everyone who works on the Creative Industry. Often, material comes from clients on lower resolutions, and it affects directly the final work. Images end up blurry, and retouching them is a manual and time consuming process.

This project aims at finding alternative methods of scaling images up, using machine learning. 

A regressor was built, using the Caltech-256 database as the training set, capable of resizing images up to 25%. 

One of the main inspirations for this project was [this](http://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf) paper.

### Problem Statement
Given a low resolution image, I want to figure out a way of resizing it that's better than *nearest neighbor*, one of the simplest, yet most common ways of rescaling bitmaps.

And *better* is defined by comparing the mean difference between pixel colors on the labels and the final predicted image.

Using Caltech's image database, a regressor will be trained to be able to predict what a bigger image would look like. This database will be split into 3 sets: Training, testing and validation. This will way we can guarantee that the model is able to generate predictions on data that it has never seen.

In the past, auto-encoders and stacked auto-encoders were used to remove noise from images, and a similar architecture seems like the obvious option to tackle this problem. 

An array of pixels will be fed into a neural net, with auto-encoder-like structure, in a manner that the output of the last layer will be another array, with images 50% bigger than the input.

To achieve these results, the following approach was used:

1. Preprocess Caltech-256 to have clearer images, with normalized data
2. Create an auto-encoder network structure
3. Tweak values, try different optimizers and tune parameters
4. Present results, comparing accuracies

### Metrics

For this problem, 2 metrics will be used. Despite being similar, there is one major difference between them that made me break them into two, as shown below:

#### Loss calculation:
To calculate the loss, and efficiently backpropagate gradients, R2 was chosen.

```python
loss = tf.reduce_mean(tf.pow(tf_y - y_pred, 2))
```

Squared difference scale errors up, meaning that higher differences will be avoided more vehemently.

#### Results comparison:
To compare results, I want to have the actual image difference, without any scaling because it will distort the true distance between all colors.

```python
np.mean(np.abs(predictions - labels))
```

The absolute value of the difference is used since the subtraction may end up with positive *or* negative values for each item in the array. Taking a mean between *-2* and *2* results in *0* instead of *2*, which is the number we are looking for.

## II. Analysis

### Data Exploration

Caltech-256 consists of *30607* images, split into *256* categories.  
It's distributed as a compressed file that can be downloaded from [here](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).

The way the images were categorized is as simple as possible, only splitting them into folders:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/caltech_samples/folders.png?raw=true" height="300">

All files also follow a single nomenclature, that can be represented as:

```python
"%03i_%04i.jpg" % (category, file_number)
```
where `category` represents one of the *256* file categories and `file_number` the index of the file, starting at *1*. 

All images use the JPEG compression format.

Since the purpose of this project is not classification, these categories were ignored and all images were shuffled. Also to constraint how much RAM will be used, a smaller set, of *20000* image is being extracted, otherwise `swap` space would be used, slowing down the whole process and `pickle` wouldn't be able to serialize those arrays into the disk, due to file size restrictions.

Image dimensions are not normalized and their resolution is also not stable. Some of them are blurry, monochromatic, and/or small. Data preprocessing is definitely a requirement to improve results.

Downscaling and cropping had to be applied in order to minimize those resolution inconsistencies, this process will be discussed in depth on the *Proprocessing* section of this document.

Also, to simplify computations, all images were converted to grayscale, having only 1 color channel.

### Exploratory Visualization

Take a look at those images, which were extracted from Caltech-256:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/caltech_samples/006_0018.jpg?raw=true" width="500" height="333">  
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/caltech_samples/019_0017.jpg?raw=true" width="165" height="253">  
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/caltech_samples/028_0020.jpg?raw=true" width="612" heigh="470">  
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/caltech_samples/124_0014.jpg?raw=true" width="215" height="142">  

As you can see, they have different dimensions, scale and color depth. There is some noise around the white part of the glove image and the camel blends with the background, as if image focus were not properly defined.

Preprocessing had to be applied to achieve a normalized dataset. All labels were resized and cropped to 150x150 pixels, and the training set to 100x100. This can also be interpreted as 33.3% smaller or 50% bigger.

### Algorithms and Techniques

This is a comprehensive list of algorithms tested and used during this experiment:

Network structure:

* Simple *fully connected* network
	* On a *fully connected* layer each perceptron on the current layer is connected to all perceptrons on the previous layer.
* *Multi layer perceptron* network, with 2 hidden layers
	* A *Multi layer perceptron* represents the core of deep learning, it's a network architecture with one or more hidden layers.
	* A hidden layer is a group of perceptrons whose both inputs and outputs do not touch the network input and output spaces.
* *Multi layer perceptron* network, with 4 hidden layers

Example of neural net structure:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/neural_net.gif?raw=true" width="200">

Layers with the following number of parameters:

* 4096
* 2048
* 1024
* 512

3 activation functions were tested:

* Sigmoid
	* A function that stretches intermediate values, reducing entropy
	* <img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/functions/sigmoid.png?raw=true" width="200">
* Tanh
	* A special version of a sigmoid that goes from -1 to 1
	* <img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/functions/tanh.gif?raw=true" width="200">
* ReLU
	* All results < 0 are set to 0, otherwise they can grow linearly until infinity
	* <img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/functions/relu.png?raw=true" width="200">

2 types of weight initialization:

* Gaussian distribution with various standard deviations
	* A Gaussian distribution represents how data is usually present on the real world, this makes it more powerful than a simple random initialization
	* <img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/functions/gaussian.png?raw=true" width="200">
* Xavier initialization
	* Is a special case of a Gaussian distribution, where the standard deviation is set to: `math.sqrt(6) / (math.sqrt(inputs_number) + math.sqrt(outputs_number))`

3 Optimizers:

* Regular Gradient Descent
	* An iterative algorithm where a gradient is calculated and optimized according to execution steps.
	* <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Gradient_descent.svg/350px-Gradient_descent.svg.png" width="200">
* Ada Delta
	* Ada Delta adds momentum to GD, but keeping track of previous `n` gradients and taking them into account while doing back propagation.
* RMS Prop
	* Similar to Ada Delta. Both algorithms aim at solving Adagrad's aggressive momentum, with <a href="http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop" target="_blank">minor differences on their equations</a>.

And the learning rate was also implemented on the following manners:

* Constant
* With exponential decay
	* Where the learning curve decreases exponentially after `n` epochs.

Different image patch sizes:

* in: whole image, out: whole image
* in: 16x16, out 32x32
* in: 8x8, out 16x16
* in: 10x10, out 15x15

Dropout:
Dropout works but ignoring and averaging part of the layer net input, this way the perceptrons have no option but to generalize the data, since inputs vary constantly.

* Dropout: keep_prob = 0.5
* Dropout: keep_prob = 0.9
* No dropout

#### Best algorithms and parameters

After a week playing with parameters and algorithms, I can say that the group that made more sense is:

* Multi layer perceptron with 2 hidden layers
	* Less than 2 hidden layers can't generalize the image. You see progress initially while training and then it gets stuck.
	* More than 2 hidden layers take too long to train and doesn't seem to progress, even after hours running.
* 2048 parameters for the first hidden layer, and 1024 for the second
	* More than this slows down the processing tremendously.
	* Less parameters saturate too fast and replicate the initial image.
* Tanh
	* Both sigmoid and ReLu were ignoring part of the color data, probably because the array was normalized to have colors between *-1 and 1*.
* Xavier initialization
	* Using only a gaussian distribution makes me have to tweak the standard deviation every time I changed the number of parameters on the network.
* RMS Prop optimizer
	* It's momentum keeps us from getting stuck into local optima, which happened constantly on AdaDelta and GradientDescent.
* Exponential learning rate.
	* Without it the learning curve gets bumpy, indicating the the model is learning and unlearning every `epoch`.
* Patch size: 10x10 -> 15x15
	* Smaller sizes won't have enough data to recreate the bigger version.
	* Bigger patches require deeper network with way more data than I had available to generate a sharper output.
* Dropout removes information randomly from the image, and since my dataset never overfits and I'm trying to recover edges, wasn't a good addition to the net.
	
To make sure the test results are reliable, three sets were created:
	* Training: with 75% of the data
	* Testing: with 12.5% of the data
	* Validation: with 12.5% of the data

### Benchmark

The main benchmark used to compare results, is the mean color difference:

```python
np.mean(np.abs(predictions - labels))
```

So my minimum required score would be:

```python
np.mean(np.abs(nn(X) - y))
```

Where:

* `X`: dataset item
* `y`: label
* ``nn``: a function that resizes a label using nearest neighbor algorithm, to match the size of `y`.

And a prediction score would be represented by:

```python
np.mean(np.abs(prediction - label))
```

This project is successful as long the the prediction score is higher than *nearest neighbor* score.


## III. Methodology

### Data Preprocessing

After decompressing the whole Caltech-256 database, I copied it into 2 folders: `data/X` and `data/y`.

`data/y` contains all images cropped and rescaled to *150x150*. This was done in order to normalize all sizes and aspect ratios. No images were distorted, all extra pixels were discarded.

`data/X` contains a copy of `data/y` where images were downscaled to match *100x100* pixels.

The next step was to load all files into memory, combining R, G and B channels into 1 luminance channel, using [Relative Luminance](https://en.wikipedia.org/wiki/Relative_luminance) algorithm.

Finally, the pixel values were normalized to range between -1 and 1, where -1 represent black and 1 is absolute white.

#### Data statistics after preprocessing:

##### Shape:
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/shape.png?raw=true" width="400">  

##### Mean, std deviation:
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/stats.png?raw=true" width="320">  

##### Standard deviation:
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/std_dev1.png?raw=true" height="250">
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/std_dev2.png?raw=true" height="250">  

#### Data sample after preprocessing:

**X** on the top, **y** on the bottom.  

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/2.png?raw=true" height="300">
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/3.png?raw=true" height="300">
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/4.png?raw=true" height="300">
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/preprocessing/5.png?raw=true" height="300">

### Implementation and Refinement

The first step of implementing this neural net was searching for references regarding similar work, and I landed on these 3 papers: [1](http://people.tuebingen.mpg.de/burger/neural_denoising/files/neural_denoising.pdf), [2](https://papers.nips.cc/paper/4686-image-denoising-and-inpainting-with-deep-neural-networks.pdf) and [3](http://www.sersc.org/journals/IJSIP/vol7_no3/14.pdf).

My first attempt was to create a single fully-connected network, and feed whole images in. Obviously it was simplistic approach so I moved on to a deeper architecture, still receiving a whole image.

After that, I moved away from a `GradientDescentOptimizer` in favor of a `RMSPropOptimizer` but that was still giving me blurry images. With better results though.

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/blurry_images.png?raw=true" height="300">  

Next step was to try to add `dropout`, but unless I used a really high `keep_prob`, higher than *0.9*, all I could get was a blur:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/with_dropout.png?raw=true" height="300">  

Trying `tanh` improved drastically my results, but as you can see it was still far from ideal. The learning rate was too high - and that's the reason why the chart is so bumpy - and weight initialization was off:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/with_tanh.png?raw=true" height="500">  

Adjusting the learning rate and weight initialization I was able to improve some results. But I didn't had moved on to *Xavier initialization* yet, I was still using a gaussian distribution, with TensorFlow's `truncated_normal`:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/tanh_tunning.png?raw=true" height="200">  

Moving on to deeper networks, they proved much harder to train, specially when paired with Stochastic Gradient Descent. This image shows a network on low training *epochs* but should be already enough to see at least shape of objects:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/deeper_net.png?raw=true" height="200">  

This is where things started to get interesting. The correct weight initialization made a whole difference on the results achieved. This is what it looks like to use `Xavier` instead of a regular fixed standard deviation:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/xavier.png?raw=true" height="200">  

The next steps were to start fine tuning parameters, and this is what took longer.  
I tried things like upscaling images using *nearest neighbor* and *bicubic interpolation* before sending them into the deep net, with interesting results I'll discuss later. I also tried the values described on the **Best algorithms and parameters** section.

After reaching the final model, it's stunning to see what kind of prediction it can make, specially when dealing with text, as you can observe the reconstruction of the number **5** on this image:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/tuning2.png?raw=true" height="200">  

Or this **B**:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/sample_1.png?raw=true" height="200">  

The final implementation code with prediction samples can be seen [here](https://github.com/lucasdupin/ml-image-scaling/blob/master/2_autoencoder.ipynb).

### Coding

Python is a pretty straightforward and semantic language, with strong package managing support. And Jupyter notebooks are a great way of explaining your thoughts, experimenting and iterating through code. On top of that a deep learning library is needed, and my initial choices were:

* Caffe
* Theano
* TensorFlow

In the end I chose TensorFlow due to its speed, flexibility and great documentation.

Also, bits of `numpy` and `sklearn` were used to preprocess, reshape and split arrays. `numpy` has great performance and simple syntax to manipulate data.

Coding a TensorFlow graph is a simple process, you need to define it using a snippet like this:

```python

graph = tf.Graph()
with graph.as_default():

    # input variables
    tf_X = tf.placeholder(tf.float32, shape=(batch_size, n_input))
    ...

    # weights / slopes
    W = {
        'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=std_dev_1)),
        ...
    }
    # biases / intercepts
    b = {
        'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=std_dev_1)),
        ...
    }
    
    # Execute sequence of encoding, decoding to
    # generate predictions
    layer_1 = activate(tf.add(tf.matmul(x, W['encoder_h1']), b['encoder_b1']))
    ...
    y_pred = activate(tf.add(tf.matmul(layer_1, W['decoder_h2']), b['decoder_b2']))
    
    # Our goal is to reduce the mean between the originals and the predictions.
    # This value is squared to be more aggressive with distant values
    loss = tf.reduce_mean(tf.pow(tf_y - y_pred, 2))
    
    # Define optimizer
    optimizer =  tf.train.RMSPropOptimizer(0.01).minimize(loss)

```

And let it run:

```python

with tf.Session(graph=graph) as session:
    
    print('Initialized')
    for step in range(30000):
        print(".", end="")
        
        batch_data, batch_labels = get_patches(X_train, y_train, batch_start)
            
        # Data passed into the session
        feed_dict = {
            tf_X : batch_data, 
            tf_y : batch_labels,
            global_step: step
        }
        
        # Run session
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

```

*Final implementation [here](https://github.com/lucasdupin/ml-image-scaling/blob/master/2_autoencoder.ipynb).*

It's a good idea to have the graph definition and session code on different Jupyter cells, to make the code simpler to understand and easier to tweak.

Also, always start simple and build on top of it. Having too many features while you start to build your algorithm will make you lost, without knowing what has good or poor performance. I had to take a couple steps back, removing early complexity, to make sure I was understanding the performance drawbacks and limits of each piece of code.

I would say that the most traumatic part of the coding process was to define all variables/matrices with their correct sizes. Changing the size of a layer requires manual changes on the several parts of the code. Probably a library like [TFLearn](http://tflearn.org/) could make this easier but I was afraid of loosing flexibility. 

Apart from that, coding was smooth, and the only challenges were to figure out the best parameter/algorithm combinations. GridSearch could be a great way of pushing this even further.

## IV. Results

Let's start by evaluating the score function. Its results vary from 0 to 1. Where 1 represents images that are completely different and 0 perfect regression.

```
when images are the same: 0.000000  
when images are completely different: 1.000000
```

Even though this model never overfits, a point had to be chosen to stop training, and it was when the learning rate saturated and loss between training epochs wasn't being reduced anymore:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/final_graph.png?raw=true" height="300"> 

It's important to note that test results may have a small difference between runs since they extract a random patch from every image on the dataset. But that difference isn't significant since it's only about 1%, a minor fluctuation.

Results achieved with the final model on the datasets:

```
Test r2: 0.064
Validation r2: 0.063
```

It's accurate to say that the final model could generate images that are 6% distant from the original label on average.

It's also accurate to say that this model is robust enough to deal with unseen data, since we splited the original dataset into 3. This guarantees that even if the validation set leaks into the model, the test set won't.

Since the model works on small patches, a final image has to be recomposed after going through the neural net. This was achieved simply by concatenating them, in a new `numpy` array. Implementation code can be found [here](https://github.com/lucasdupin/ml-image-scaling/blob/master/3_results.ipynb).

These are predictions of the final model:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/final_predictions.png?raw=true" height="400">  
<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/final_predictions2.png?raw=true" height="400">

It's easy to note how close the **labels** row is from the **predicted**.

And this is a score comparison after reconstruction:

<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/final_recomposition.png?raw=true" height="1000">

Comparing the *nearest neighbor* score: 0.120 with our model: 0.092 we can say there's a ~30% gain on this result.

```
0.120/0.092 = 1.3043...
```

I encourage you to load the [results notebook](https://github.com/lucasdupin/ml-image-scaling/blob/master/3_results.ipynb) and try it too, cases where a *nearest neighbor* beats this *model* are extremely rare.

### Extra, intriguing results

As part of tweaking and tuning parameters, one of the things tried was to upscale images, using *bicubic interpolation* before feeding them into the network.  
This results in a regular autoencoder, where the number of inputs on the network is the same as the number of output parameters.

Even though it defeats the purpose or creating a network that rescales images, interesting behaviors could be observed.

The average difference between images was lower, having final score of `0.04`, but even with a better score, the difference from the dataset and its labels couldn't be beaten.

But the most intricate part is that noise was removed without generating an image that's blurrier than the input. Take a look at this sample and compare the white parts. Notice how it's way less noisier:

<table>
<tr>
	<td>Data upscaled</td>
	<td>Prediction</td>
	<td>Original</td>
</tr>
<tr>
<td>
	<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/pre_scale_bicubic.png?raw=true" height="300">
</td><td>
	<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/pre_scale_model.png?raw=true" height="300">
</td><td>
	<img src="https://github.com/lucasdupin/ml-image-scaling/blob/master/project_material/pre_scale_original.png?raw=true" height="300">
</td>
</tr>
</table>

This means that alternate forms of this model can be used as an image pass, to clean up images, potentially using a mask, like a Photoshop tool.

More details and implementation can be found [here](https://github.com/lucasdupin/ml-image-scaling/blob/feature/pre_rescale/3_results.ipynb). Same repository but on a separate branch.

## V. Conclusion

Finding the correct metric is not an easy task and even though pixel difference sounds like the most obvious solution, it can't account for low quality images found on Caltech-256. Sometimes lower scores can be observed on noisy images, since the model try to clean them up strengthening/keeping edges. A better dataset could potentially improve the model.

ImageNet was also considered but a zip file containing only image URLs already is as big as 334MB, wouldn't be practical enough for a Python Notebook. Apart from that, their website was offline on the last week, so downloading it wasn't even an option.

Some lessons could be learned during this project, but the hardest one wasn't regarding machine learning, but its ecosystem.

Training image dataset on the CPU is impractical. It's not fast enough to tweak parameters without making assumptions about the learning curve. Assumptions that may not be correct. Trying to move on to GPU, too many obstacles were found and it took me about a week to overcome them.

The graphics card manufacturer and model define if you can or not run things on the GPU, since TensorFlow requires CUDA. And the main supported operating system, Linux, is known for having compatibility issues with proprietary graphics drivers. This is where I entered an infernal loophole. Latest versions of Ubuntu were not supported by NVidia yet, but GTX-1060 requires updated drivers and CUDA 8.0 but TensorFlow is built against CUDA 7.5. The final solution was to downgrade the operating system and compile TensorFlow from source, but that still left me with inconsistencies between `pickle` on macOS and Linux.

### Improvement

Fliboard [posted on their blog](http://engineering.flipboard.com/2015/05/scaling-convnets/) a similar experiment where they use Convnets instead of auto-encoders to do this kind image processing.  
Despite using images of way better quality and a dataset that's 200 times bigger than Caltech-256 -- with a total of 3 million images --, they introduced 2 convolution layers on the entry point of their DNN.  
This technique improves edge and feature detection and is something I definitely want to try as soon as I have some spare time.

Another strategy to optimize results could be to export overlapping patches and recompose them using a gaussian/normal distribution, this would avoid some noticeable seams on the current predictions.
