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
Given a low resolution image, I want to figure out a way of resizing it that's at least better than *nearest neighbor*, one of the simplest, yet most common ways of rescaling bitmaps.

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

<img src="https://github.com/lucasdupin/enhance/blob/master/project_material/caltech_samples/folders.png?raw=true">

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

These are samples of those images:

<img src="https://github.com/lucasdupin/enhance/blob/master/project_material/caltech_samples/006_0018.jpg?raw=true">  
<img src="https://github.com/lucasdupin/enhance/blob/master/project_material/caltech_samples/019_0017.jpg?raw=true">  
<img src="https://github.com/lucasdupin/enhance/blob/master/project_material/caltech_samples/028_0020.jpg?raw=true">  
<img src="https://github.com/lucasdupin/enhance/blob/master/project_material/caltech_samples/124_0014.jpg?raw=true">  

As you can see, the have different dimensions, scale and color depth.

Preprocessing had to be applied to achieve a normalized dataset. All labels were resized to 150x150 pixels, and the training set to 100x100. This can be written as 33.3% smaller or 50% bigger.


In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
25% vs 50%
rescaled vs smaller counterpart
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
* Caltech256 database
* Resized images because some of them were blurred
* Cropped them to normalize their size
* Created smaller versions, downscaling images by 50%: having 1/4th of the pixels. Also tried 33% but the net had about the same final score 
* Normalized data, to have them go from -1 to 1, having a mean close to 0
* Created train, test and validation datasets, composed of 15k, 2.5k and 2.5k images respectively

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
* Loss function is R2
* Metrics for comparing images: like r2, but without squaring it, only the plain difference between images
* RMSProp optimizer
* Structure similar to an auto-encoder
* Activations is a tanh, like a sigmoid, but going from -1 t 1, like the data
* Small 10x10 patches outputs 20x20 images

In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
* Xavier initialization
* Patch size: big patches are blurrier, small have recomposition problems
* Learning rate: bumpy chart, unlearns
* Number of hidden layers: more than 3 makes it harder to learn
* Number of features: 512 won't do a good job, more than 2048 won't make a difference and slows it down - at least on my setup
* Optimizers: regular gradient descent seems to get stuck

In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
