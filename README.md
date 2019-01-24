# NeuralGenetic
This project optimizes the **artificial neural network (ANN)** parameters using the **genetic algorithm (GA)** for the classification of the Fruits360 dataset. The implementation is from scratch using **NumPy**.  

## Part 1
This project is an extension to a previous project which is documented in a tutorial titled **Artificial Neural Network Implementation using NumPy and Classification of the Fruits360 Image Dataset** which is available at my **LinkedIn profile** here: https://www.linkedin.com/pulse/artificial-neural-network-implementation-using-numpy-fruits360-gad. It starts by extracting the image features from 4 selected classes of the Fruits360 dataset which are Apple Braeburn,	Lemon Meyer, Mango,	and Raspberry. The feature vector consisted of 360 bin histogram of the Hue channel. 

The feature vector is reduced by filtering it using the standard deviation to return a new vector of length 102. The reduced feature vector is used for training the ANN that is implemented from scratch using NumPy. The implementation of the first tutorial is available at this link at my **GitHub page**: https://github.com/ahmedfgad/NumPyANN  

In the previous project, the ANN was not completely created as just the forward pass was made ready but there is no backward pass for updating the network weights. This is why the accuracy is very low and not exceeds 45%. The solution to this problem is using an optimization technique for updating the network weights. This project uses the genetic algorithm (GA) for optimizing the network weights.

It is worth-mentioning that both the previous and this tutorial are based on my 2018 book cited as **Ahmed Fawzy Gad 'Practical Computer Vision Applications Using Deep Learning with CNNs'. Dec. 2018, Apress, 978-1-4842-4167-7**. The book is available at Springer at this link: **https://springer.com/us/book/9781484241660**. You can find all details within this book.  

## Part 2
The extension to the previous tutorial, which is implemented by this project, consists of 3 main Python files which are **ANN.py**, **GA.py**, and **Example_GA_ANN.py** which is the main file from which the other files are imported and called. This file uses 2 supplementary files which are the previously extracted dataset features stored into a file named **dataset_features.pkl**. The second file is the class labels for all samples which are stored into a file named **outputs.pkl**.

The **Example_GA_ANN.py** file reads the features and the class labels files, filters the features based on the standard deviation, creates the ANN architecture, generates the initial solutions, loops through a number of generations by calculating the fitness values for all solutions, selecting best parents, applying crossover and mutation, and finally creating the new population.

The documentartion of the second tutorial is available at my **LinkedIn profile** here: https://www.linkedin.com/pulse/artificial-neural-networks-optimization-using-genetic-ahmed-gad

## Read more about GA
Before going further in this project, I recommend reading about the GA and its implementation in Python from scratch. You can find details it **my book** cited above. Also you can read my previous tutorials found at these links:
* **Introduction to Optimization with Genetic Algorithm**  
https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad  
https://www.kdnuggets.com/2018/03/introduction-optimization-with-genetic-algorithm.html  
https://towardsdatascience.com/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b  

* **Genetic Algorithm (GA) Optimization - Step-by-Step Example**  
https://www.slideshare.net/AhmedGadFCIT/genetic-algorithm-ga-optimization-stepbystep-example

* **Genetic Algorithm Implementation in Python**  
https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad  
https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html  
https://towardsdatascience.com/genetic-algorithm-implementation-in-python-5ab67bb124a6  
https://github.com/ahmedfgad/GeneticAlgorithmPython  

## For contacting the author  
* E-mail: ahmed.f.gad@gmail.com
* LinkedIn: https://linkedin.com/in/ahmedfgad/
* KDnuggets: https://kdnuggets.com/author/ahmed-gad
* YouTube: https://youtube.com/AhmedGadFCIT
* TowardsDataScience: https://towardsdatascience.com/@ahmedfgad
* GitHub: https://github.com/ahmedfgad
