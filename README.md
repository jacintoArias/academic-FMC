# Factor-Based Multilabel Classifier

##### Copyright 2015 Jacinto Arias - www.jarias.es

This package contains an experimental implementation of the method presented in:


##### J. Arias, J.A. Gamez, T.D. Nielsen and J.M. Puerta 
#### "A scalable pairwise class interaction framework for multidimensional classification."

Please visit the companion website of the paper for more information:
http://simd.albacete.org/supplements/FMC.html

---------------------------------------------------------------------------------------

##### Description

This software implements the FMC algorithm. As it is only a prototype several foreign libraries and packages are used:

- **UGM** : Mark Schmidt implementation of undirected graphical models in Matlab: http://www.cs.ubc.ca/~schmidtm/Software/UGM.html
- **Weka** : Data Mining platform from the University of Waikato: https://weka.wikispaces.com
- **Mulan** : Library for multilabel extension of the Weka library: http://mulan.sourceforge.net

Our program is split in two platforms Java and Matlab, so please confirm that you have both installed before configuring the experiments.

Right now we only provide running scripts for unix based systems, but windows script and more flexibility will be added shortly. Do not hesitate to contact us for any matter, we will be glad to help.


------------------------------

##### Disclaimer

This software is released for scientific purpose only, use it at you own risk. If you plan to use it professionally please contact us for further support.


------------------------------

##### Instructions

To run the software please use the `run_unix.sh` script.


1) First change the matlab path by your own system path by modifying the appropiate line.


2) The script needs two parameters:
    `-arff <filename>`
    `-xml <filename>`
    
These files correspond with the input data format defined for the Mulan library, please check their
website for additional information. Example input data can be found in the "./data" folder to test
the program.
    
    Examples:
    
    `$./run_unix.sh -arff ./data/emotions.arff -xml ./data/emotions.xml`
    `$./run_unix.sh -arff ./data/scene.arff -xml ./data/scene.xml`


3) The software is prepared to run just by launching the script if the folder structure has been
maintained. Please let us know if you experiment any bug at this time (Pardon for the inconveniences).


4) The script accepts additional parameters to configure the experiment and the FMC algorithm:

    `-classifier : Default "bayes.NaiveBayes" . Any weka classifier class can be included here by specifying the class path.`
    
    `-discretization |supervised|unsupervised| : Default "supervised" . The framework supports "supervised" and "unsupervised" discretizations.`
    
    `-fss |CFS|no| : Default "CFS" . The framework can be configured to perform attribute selection "CFS" or not "no"`
         
    `-cv : Default 10. Numeric parameter to specify the number of folds for the crossvalidation.`
    
    `-seed : Default 1990. Numeric parameter to specify the random seed used by the problem.`


------------------------------

##### Structure of the Directory:

The root folder contains:

- **run_unix.sh** : Main script to launch the software.

- **README.txt** : This file.

- **license.txt** : GPL distribution license.

- **src/** : Folder with source and executable code.
    - **src/FMC.jar** : Java classes with the algorithm.
    - **src/MRFInference** : Matlab function with se second step of the algorithm.
    - **src/java** : java source files.
    
- **lib/** : Folder with needed foreign packages.
    - **lib/UGM** : Mark Schmidt implementation of undirected models in Matlab: http://www.cs.ubc.ca/~schmidtm/Software/UGM.html
    - **lib/weka-3.7.6.jar** : Weka distribution from University of Waikato: https://weka.wikispaces.com
    - **lib/mulan.jar** : Mulan library for multilabel extension of the Weka library : http://mulan.sourceforge.net
    
- **data/** : Example data files taken from the Mulan repository: http://mulan.sourceforge.net/datasets-mlc.html

- **temp_files** : empty folder used to store temporary files from the execution.


