/**

Copyright 2015 Jacinto Arias  - www.jarias.es

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.



FMC multilabel classifier. This source code is an experimental
implementation of the method presented in:

J. Arias, J.A. Gamez, T.D. Nielsen and J.M. Puerta 

A scalable pairwise class interaction framework for multidimensional classification.

**/



package es.jarias.FMC;

import java.util.Random;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;
import weka.core.Utils;

/**
 * 
 * @author jarias
 *
 */
public class RunFMC {

	public static void main(String[] args) {

		try {
			
			String classifierName = Utils.getOption("classifier", args);
			
			String discType = Utils.getOption("disc", args);
			String fss = Utils.getOption("fss", args);
			String prune = Utils.getOption("prune", args);
			
			String arffFilename = Utils.getOption("arff", args);
			String xmlFilename = Utils.getOption("xml", args);
			String outPath = Utils.getOption("outpath", args);
			
			String cvString = Utils.getOption("cv", args);
			String seedString = Utils.getOption("seed", args);
			
			// Check parameters:
			if (arffFilename.equals("") || arffFilename.equals("") || arffFilename.equals(""))
				throw new Exception("Please provide valid input and output files.");
			
			// Set Defaults:
			if (classifierName.equals(""))
				classifierName = "bayes.NaiveBayes";
			
			if (discType.equals(""))
				discType = "supervised";
			
			if (fss.equals(""))
				fss = "CFS";
			
			if (prune.equals(""))
				prune = "full";
			
			int cv_folds = 10;
			if (!cvString.equals(""))
				cv_folds = Integer.parseInt(cvString);
			
			int seed = 1990;
			if (!seedString.equals(""))
				seed = Integer.parseInt(seedString);
			
			MultiLabelInstances original = null;
			try {
				original = new MultiLabelInstances(arffFilename, xmlFilename);
			} catch (InvalidDataFormatException e)
			{
				System.out.println("Please provide valid multilabel arff+xml mulan files");
				System.exit(-1);
			}
			
			MultiLabelInstances dataset = original.clone();
			Instances aux = dataset.getDataSet();
			aux.randomize(new Random(seed));
		
			dataset = dataset.reintegrateModifiedDataSet(aux);
			
			System.out.println("--------------------------------------------");
			System.out.println("FMC multi-label classifier experiment");
			System.out.println("-Pruning strategy: "+prune);
			System.out.println("-Base Classifier: "+classifierName);
			System.out.println("-Discretization: "+discType);
			System.out.println("-Feature Selection: "+fss);
			System.out.println("-Folds: "+cv_folds);
			System.out.println("-Seed: "+seed);
			
			// Perform CV or Holdout
			if (cv_folds != 0)
			{
				for (int fold = 0; fold < cv_folds; fold++)
				{
					MultiLabelInstances trainData = original.reintegrateModifiedDataSet(dataset.getDataSet().trainCV(cv_folds, fold));
					MultiLabelInstances testData  = original.reintegrateModifiedDataSet(dataset.getDataSet().testCV(cv_folds, fold));
					
					FMC.buildModel(trainData, testData, fold, classifierName, discType, fss, outPath, prune);
				}
			}
			else
			{
				double HOLDOUT_PERCENTAGE = 0.6;
				
				int trainSize = (int) Math.floor(dataset.getNumInstances() * HOLDOUT_PERCENTAGE);
				int testSize = dataset.getNumInstances() - trainSize;
					
				MultiLabelInstances trainData = dataset.reintegrateModifiedDataSet(new Instances(dataset.getDataSet(), 0, trainSize));
				MultiLabelInstances testData  = dataset.reintegrateModifiedDataSet(new Instances(dataset.getDataSet(), trainSize, testSize));
				
				FMC.buildModel(trainData, testData, 0, classifierName, discType, fss, outPath, prune);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
