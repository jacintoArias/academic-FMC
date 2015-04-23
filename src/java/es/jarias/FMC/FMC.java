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

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 * 
 * @author jarias
 *
 */
public class FMC {

	public static void buildModel(MultiLabelInstances trainData, MultiLabelInstances testData, int fold, String baseClassifierClass, 
			String discType, String fss, String outPath, String prune) 
			throws Exception {
		
		double start = System.nanoTime();
		
		try
		{
		
		// DATA PREPROCESING:
			
		weka.filters.unsupervised.attribute.Discretize m_unsuperDiscretize = null;
		
		if (discType.equals("supervised"))
		{
			// pass
			// Supervised discretization is applied to each model later during the training step.
		}
		else if (discType.equals("unsupervised"))
		{
			// Apply a baseline discretization filter:
			 m_unsuperDiscretize = new weka.filters.unsupervised.attribute.Discretize();
			 m_unsuperDiscretize.setUseEqualFrequency(false);
			 m_unsuperDiscretize.setBins(3);
			 m_unsuperDiscretize.setInputFormat(trainData.getDataSet());
			
			trainData = trainData.reintegrateModifiedDataSet(Filter.useFilter(trainData.getDataSet(), m_unsuperDiscretize));
		}
		else
				throw new Exception("Invalid Discretization Type");
			
		if (!fss.equals("no") && !fss.equals("CFS"))
			throw new Exception("Invalid FSS strategy");
		
		if (!prune.equals("full") && !prune.equals("tree") && !prune.equals("best") && !prune.equals("hiton")  && !prune.equals("bdeu"))
			throw new Exception("Invalid Pruning strategy");
		
		
		// Label information
		int 	 m_numLabels = trainData.getNumLabels();
		int[]	 m_labelIndices = trainData.getLabelIndices();
			
		// Map for reference:
		HashMap<Integer, Integer> mapLabels = new HashMap<Integer, Integer>(m_numLabels);
		String[] mapLabelsName = new String[m_numLabels];
		for (int l = 0; l < m_numLabels; l++)
		{
			mapLabels.put(trainData.getLabelIndices()[l], l);
			mapLabelsName[l] = trainData.getDataSet().attribute(trainData.getLabelIndices()[l]).name();
		}
			
		// Get label combinations:
		int		m_numPairs = ( m_labelIndices.length * (m_labelIndices.length - 1) ) / 2;
		int[][]	labelCombinations = new int[m_numPairs][2];
		
		int counter = 0;
		for (int i = 0; i < m_labelIndices.length; i++)
		{
			for (int j = i+1; j < m_labelIndices.length; j++)
			{
				labelCombinations[counter] = new int[]{m_labelIndices[i],m_labelIndices[j]};
				counter++;
			}
		}
			
		
		// Select the pairs:
		int m_numSelected = m_numPairs;
		int m_numSingleton = 0;
		int[] ordered;
		boolean[] selectedPair = new boolean[m_numPairs];
		boolean[] singleton = new boolean[m_numLabels];
		
		for (int i = 0; i < m_numPairs; i++)
			selectedPair[i] = true;
			
		if (!prune.equals("full"))
		{
			
			m_numSelected = 0;
			selectedPair = new boolean[m_numPairs];
			
			// Info gain for pruned model:
			double[][] mutualInfoPairs = mutualInfo(trainData.getDataSet(), trainData.getLabelIndices());
			double[] mutualInfo = new double[m_numPairs];
			counter = 0;
			for (int i = 0; i < m_labelIndices.length; i++)
			{
				Instances tempInstances = new Instances(trainData.getDataSet());
				tempInstances.setClassIndex(m_labelIndices[i]);
				
				for (int j = i+1; j < m_labelIndices.length; j++)
				{
					mutualInfo[counter] = mutualInfoPairs[i][j];
					counter++;
				}
			}
			
			ordered = orderBy(mutualInfo);
			
			if (prune.equals("tree"))
			{
				// Each labels correspond to its own connex component 
				HashMap<Integer, ArrayList<Integer>> tree_compo = new HashMap<Integer, ArrayList<Integer>>(m_numLabels);
				HashMap<Integer, Integer> tree_index = new HashMap<Integer, Integer>(m_numLabels);
				
				for (int i = 0; i < m_numLabels; i++)
				{
					tree_compo.put(i, new ArrayList<Integer>());
					tree_compo.get(i).add(i);
					tree_index.put(i, i);
				}
					
				for (int i = 0; i < m_numPairs; i++)
				{
					if (m_numSelected >= m_numLabels-1)
						break;
					
					int pairIndex = ordered[i];
					int pair_i = mapLabels.get(labelCombinations[pairIndex][0]);
					int pair_j = mapLabels.get(labelCombinations[pairIndex][1]);
					
					int conex_i = tree_index.get(pair_i);
					int conex_j = tree_index.get(pair_j);
					
					if (conex_i != conex_j)
					{
						ArrayList<Integer> family = tree_compo.get(conex_j);
						tree_compo.get(conex_i).addAll(family);
						for (int element : family)
						{
							tree_index.put(element, conex_i);
						}
						
						selectedPair[pairIndex] = true;
						m_numSelected++;
					}
				}
			} // End of the chow-liu algorithm
			
			if (prune.equals("best") || prune.equals("tree"))
			{
				int amount = 0;
				if (prune.equals("best"))
					amount = (int) (m_numLabels * 2);
				
				int index = 0;
				while(m_numSelected < amount && index < m_numPairs)
				{
					if (!selectedPair[ordered[index]])
					{
						m_numSelected++;
						selectedPair[ordered[index]] = true;
					}
					
					index++;
				}
			}// End of the linear tree and best procedures
				
			if (prune.equals("hiton"))
			{
				weka.filters.unsupervised.attribute.Remove m_remove = new weka.filters.unsupervised.attribute.Remove(); 
				m_remove.setAttributeIndicesArray(trainData.getLabelIndices());
				m_remove.setInvertSelection(true);
				m_remove.setInputFormat(trainData.getDataSet());
				Instances hitonData = Filter.useFilter(trainData.getDataSet(), m_remove);
				
				HITON hiton = new HITON(hitonData);
				
				HashSet<Integer>[] markovBlanket = new HashSet[m_numLabels];
				for (int l = 0; l < m_numLabels; l++)
					markovBlanket[l] = hiton.HITONMB(l);
				
				for (int p = 0; p < m_numPairs; p++)
				{
					int p_i =  mapLabels.get(labelCombinations[p][0]);
					int p_j =  mapLabels.get(labelCombinations[p][1]);
				
					if (markovBlanket[p_i].contains(p_j) || markovBlanket[p_j].contains(p_i))
					{
						selectedPair[p] = true; 
						m_numSelected++;
					}
				}
				
			} // end of the hiton pruning algorithm
			
			if (prune.equals("bdeu"))
			{
				weka.filters.unsupervised.attribute.Remove m_remove = new weka.filters.unsupervised.attribute.Remove(); 
				m_remove.setAttributeIndicesArray(trainData.getLabelIndices());
				m_remove.setInvertSelection(true);
				m_remove.setInputFormat(trainData.getDataSet());
				Instances hitonData = Filter.useFilter(trainData.getDataSet(), m_remove);
				
				BDeu hiton = new BDeu(hitonData);
				double [] scores = hiton.singleScore;
				
				double[] pairScores = new double[m_numPairs];
				double[] sumScores = new double[m_numLabels];
				for (int p = 0; p < m_numPairs; p++)
				{
					int head = mapLabels.get(labelCombinations[p][0]);
					int tail = mapLabels.get(labelCombinations[p][1]);
					pairScores[p] = -1*(scores[tail] - (hiton.localBdeuScore(tail, new Integer[]{head})));
					
					sumScores[tail] += pairScores[p];
					sumScores[head] += pairScores[p];
				}

				
				HashSet<Integer>[] parents = new HashSet[m_numLabels];
				for (int i = 0; i < m_numLabels; i++)
					parents[i] = new HashSet<Integer>();
				
				ordered = orderBy(pairScores);
				
				int[] topologicalOrdering = orderBy(sumScores);
				
				int[] relevance = new int[m_numLabels];
				for (int i = 0; i < m_numLabels; i++)
					relevance[topologicalOrdering[i]] = i;
				
				for (int p = 0; p < m_numPairs; p++)
				{
					int pair = ordered[p];
					
					int head = mapLabels.get(labelCombinations[pair][0]);
					int tail = mapLabels.get(labelCombinations[pair][1]);

					if (relevance[head] > relevance[tail])
					{
						int aux = head;
						head = tail;
						tail = aux;
					}
					
					// Check if adding this improves
					parents[tail].add(head);
					double scoreAdd = hiton.localBdeuScore(tail, parents[tail].toArray(new Integer[parents[tail].size()]));
					double diff = scores[tail] - scoreAdd;
					
					if (diff < 0)
					{
						scores[tail] = scoreAdd;
						selectedPair[pair] = true; 
						m_numSelected++;
					}
					else
					{
						parents[tail].remove(head);
					}
				}// End of the BDeu procedure
				
			}// End of the Pruning algorithms 
			
			//
			// Determine singleton variables
			for (int i = 0; i < m_labelIndices.length; i++)
				singleton[i] = true;
				
			for (int p = 0; p < m_numPairs; p++)
			{
				if (selectedPair[p])
				{
					singleton[mapLabels.get(labelCombinations[p][0])] = false;
					singleton[mapLabels.get(labelCombinations[p][1])] = false;
				}
			}
				
			for (int i = 0; i < m_labelIndices.length; i++)
				if (singleton[i])
					m_numSingleton++;
			
			mutualInfo = null;
		}
		
		// Generate single class datasets from the full ML data and learn models:
		HashMap<Integer, Classifier> models = new HashMap<Integer, Classifier>();
		HashMap<Integer, Classifier> singletonModels = new HashMap<Integer, Classifier>();
		HashMap<Integer, weka.filters.supervised.attribute.AttributeSelection> singletonFilterSel = new HashMap<Integer, weka.filters.supervised.attribute.AttributeSelection>();
		HashMap<Integer, weka.filters.supervised.attribute.Discretize> singletonFilter = new HashMap<Integer, weka.filters.supervised.attribute.Discretize>();
		weka.filters.supervised.attribute.AttributeSelection[] m_selecters = new weka.filters.supervised.attribute.AttributeSelection[m_numPairs];
		weka.filters.supervised.attribute.Discretize[] m_discretizers = new weka.filters.supervised.attribute.Discretize[m_numPairs];
		
		
		ClassCompoundTransformation[] converters	= new ClassCompoundTransformation[m_numPairs];
			
		for (int i = 0; i < m_numPairs; i++)
		{
			
			if (!selectedPair[i])
			{
				continue;
			}
			
			MultiLabelInstances filteredLabelData = trainData.reintegrateModifiedDataSet(RemoveAllLabels.transformInstances(trainData.getDataSet(), complement(m_labelIndices, labelCombinations[i])));
			
			converters[i] = new ClassCompoundTransformation();
			
			Instances singleLabelData = converters[i].transformInstances(filteredLabelData);
			
			if (discType.equals("supervised"))
			{
				m_discretizers[i] = new Discretize();
				m_discretizers[i].setInputFormat(singleLabelData);
				singleLabelData = Filter.useFilter(singleLabelData, m_discretizers[i]);
			}
			
			if (fss.equals("CFS"))
			{
				
				m_selecters[i] = new weka.filters.supervised.attribute.AttributeSelection();
				m_selecters[i].setSearch(new weka.attributeSelection.BestFirst());
				m_selecters[i].setEvaluator(new weka.attributeSelection.CfsSubsetEval());
				m_selecters[i].setInputFormat(singleLabelData);
				singleLabelData = Filter.useFilter(singleLabelData, m_selecters[i]);

			} 
			
			models.put(i, (Classifier) Class.forName("weka.classifiers."+baseClassifierClass).newInstance());
			models.get(i).buildClassifier(singleLabelData);
		}
		
		// Learn singleton models:
		for (int i = 0; i < m_labelIndices.length; i++)
		{
			if (singleton[i])
			{
				
				Instances singleLabelData = new Instances(trainData.getDataSet());
				singleLabelData.setClassIndex(m_labelIndices[i]);
				singleLabelData = RemoveAllLabels.transformInstances(singleLabelData, complement(m_labelIndices, new int[]{m_labelIndices[i]}));
				
				if (discType.equals("supervised"))
				{
					singletonFilter.put(i, new Discretize());
					singletonFilter.get(i).setInputFormat(singleLabelData);
					singleLabelData = Filter.useFilter(singleLabelData, singletonFilter.get(i));
				}
				
				if (fss.equals("CFS"))
				{
					weka.filters.supervised.attribute.AttributeSelection tempFilter = new weka.filters.supervised.attribute.AttributeSelection();
					tempFilter.setSearch(new weka.attributeSelection.BestFirst());
					tempFilter.setEvaluator(new weka.attributeSelection.CfsSubsetEval());
					tempFilter.setInputFormat(singleLabelData);
					singletonFilterSel.put(i, tempFilter);
					singleLabelData = Filter.useFilter(singleLabelData, singletonFilterSel.get(i));
				}

					
				Classifier single;
						
				single = (Classifier) Class.forName("weka.classifiers."+baseClassifierClass).newInstance();
				
				single.buildClassifier(singleLabelData);
				singletonModels.put(i, single);
			}
		}
			
		
		//
		// END OF THE LEARNING STAGE
		//
		
		double train = System.nanoTime() - start;
		start = System.nanoTime();
		
		Writer writerConf = null;
		Writer writerDist = null;
		Writer writerSing = null;
		Writer writerLayo = null;
			
		try 
		{
			
		    writerConf = new BufferedWriter(new OutputStreamWriter(
		    new FileOutputStream(outPath+"/conf_"+fold+".txt"), "utf-8"));
		    
		    writerDist = new BufferedWriter(new OutputStreamWriter(
		    new FileOutputStream(outPath+"/dist_"+fold+".txt"), "utf-8"));
		    
		    writerSing = new BufferedWriter(new OutputStreamWriter(
		    new FileOutputStream(outPath+"/sing_"+fold+".txt"), "utf-8"));
		    
		    writerLayo = new BufferedWriter(new OutputStreamWriter(
		    new FileOutputStream(outPath+"/layo_"+fold+".txt"), "utf-8"));
		    for (int l = 0; l < m_numLabels; l++)
		    {
		    	writerLayo.write(trainData.getDataSet().attribute(m_labelIndices[l]).numValues()+"\t");
		    }
		    writerLayo.write("\n");
		    writerLayo.write(m_numSelected+"\t"+m_numSingleton);
		    writerLayo.close();
		    
		    
		    
			// Get distributions for instance for each variable pairs:
			double[] distributions;
			
			for (int i = 0; i < testData.getDataSet().size(); i++)
			{
				
				for (int l : testData.getLabelIndices())
					writerConf.write((int) testData.getDataSet().instance(i).value(l)+"\t");
				
				writerConf.write("\n");
				
				Instance inst = testData.getDataSet().get(i);
				
				if (discType.equals("unsupervised"))
				{
					m_unsuperDiscretize.input(inst);
					inst = m_unsuperDiscretize.output();
				}
				
				for (int p = 0; p < m_numPairs; p++)
				{
					if (!selectedPair[p])
					{
						continue;
					}
					
					Instance processed = converters[p].transformInstance(inst, testData.getLabelIndices());
					
					if (discType.equals("supervised"))
					{
						m_discretizers[p].input(processed);
						processed = m_discretizers[p].output();
	
//						m_removers[p].input(processed);
//						processed = m_removers[p].output();
					}

					if (!fss.equals("no"))
					{
						m_selecters[p].input(processed);
						processed = m_selecters[p].output();
					}
					
					distributions = models.get(p).distributionForInstance(processed);
					
					writerDist.write(mapLabels.get(labelCombinations[p][0])+"\t"+mapLabels.get(labelCombinations[p][1])+"\t");
					
					for (int d = 0; d < distributions.length; d++)
						writerDist.write(distributions[d]+"\t");
					
					writerDist.write("\n");
				}
				
				
				// Get predictions for singleton labels:
				for (int m = 0; m < m_labelIndices.length; m++)
				{
					if (singleton[m])
					{
						Instance processed = RemoveAllLabels.transformInstance(inst, complement(m_labelIndices, new int[]{m_labelIndices[m]}));
						
						if (discType.equals("supervised"))
						{
							singletonFilter.get(m).input(processed);
							processed = singletonFilter.get(m).output();
						}

						if (!fss.equals("no"))
						{
							singletonFilterSel.get(m).input(processed);
							processed = singletonFilterSel.get(m).output();
						}
						
						
						double[] distribution = singletonModels.get(m).distributionForInstance(processed);
						
						double maxValue = 0;
						int conf = -1;
						
						for (int v = 0; v < distribution.length; v++)
						{
							if (distribution[v] > maxValue)
							{
								maxValue = distribution[v];
								conf = v;
							}
						}
						writerSing.write(i+"\t"+m+"\t"+conf+"\n");
					}
				}
			}
			
			writerConf.close();
			writerDist.close();
			writerSing.close();
			
			double test = System.nanoTime() - start;
			
//			train /= 1000000000.0;
//			test /=  1000000000.0;
//			System.out.println(java.lang.String.format("FMC-%s\t%s\t%s\t%d\t%s\t%s\t%.4f\t%.4f",prune,baseClassifierClass,dbName,fold,discType,fss,train,test));
		} 
		catch (IOException ex)
		{
		  // report
		} finally {
		   try {writerConf.close();} catch (Exception ex) {}
		   try {writerDist.close();} catch (Exception ex) {}
		}
		
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/* Auxiliary methods */

	public static int[] complement(int[] vector, int[] comp)
	{	
		ArrayList<Integer> result = new ArrayList<Integer>();
		
		for (int i = 0; i < vector.length; i++)
		{
			boolean contained = false;
			for (int e : comp)
			{
				if (vector[i] == e)
				{
					contained = true;
					break;
				}
			}
			if (!contained)
				result.add(vector[i]);
		}
		
		int[] res = new int[result.size()];
		
		for (int i = 0; i < result.size(); i++)
			res[i] = result.get(i);
			
		return res;
	}

	
	public static int[] orderBy(double[] values)
	{

		ArrayList<Integer> aux = new ArrayList<Integer>();
		
		for (int i = 0; i < values.length; i++)
		{
			for (int j = 0; j < values.length; j++)
			{
				if ( aux.size() <= j || values[aux.get(j)] < values[i])
				{
					aux.add(j, i);
					break;
				}
			}
		}
		
		int [] array = new int[values.length];
		for (int i = 0; i < values.length; i++)
			array[i] = aux.get(i); 
			
		return array;
	}
	
	public static double[][] mutualInfo(Instances data, int[] indexes)
	{
		
		double[][]		m_counts 		= new double[indexes.length][];
		double[][][] 	m_2counts 		= new double[indexes.length][indexes.length][];
		
		double[]		nValues 		= new double[indexes.length];
		
		double[][] I = new double[indexes.length][indexes.length];
		
		for (int i = 0; i < indexes.length; i++)
		{
			nValues[i] = data.attribute(indexes[i]).numValues();
			m_counts[i] = new double[(int) nValues[i]];
		}
		
		for (int i = 0; i < indexes.length; i++)
		{
			for (int j = 0; j < indexes.length; j++)
			{
				if (i != j)
				{
					double cardinality = nValues[i] * nValues[j]; 
					m_2counts[i][j] = new double[(int) cardinality];
				}
			}
		}
		
		// Compute counts:
		for (Instance d : data)
		{
			for (int i = 0; i < indexes.length; i++)
			{
				m_counts[i][(int) d.value(indexes[i])]++;
				for (int j = 0; j < indexes.length; j++)
				{
					if (i != j)
					{
						int index = (int) (d.value(indexes[j]) * nValues[i] + d.value(indexes[i]));
						m_2counts[i][j][index]++;
					}
				}
			}
		}
		
		
		// Calculate MI(X_i; X_j)
		for (int i = 0; i < indexes.length; i++)
		{
			for (int j = 0; j < indexes.length; j++)
			{
				if (i != j)
				{
					double mi = 0.0;
					for (int v_i = 0; v_i < nValues[i]; v_i++)
					{
						for (int v_j = 0; v_j < nValues[j]; v_j++)
						{
							
							if ((1.0 * data.numInstances() * m_2counts[i][j][(int) (v_j * nValues[i] + v_i)]) / (1.0 * m_counts[i][v_i] * m_counts[j][v_j]) > 0)
								mi += m_2counts[i][j][(int) (v_j * nValues[i] + v_i)] * Math.log((1.0 * data.numInstances() * m_2counts[i][j][(int) (v_j * nValues[i] + v_i)]) / (1.0 * m_counts[i][v_i] * m_counts[j][v_j]));
						}
					}
					I[i][j] = mi / data.numInstances();
				}
			}
		}
		
		return I;
	}
}
