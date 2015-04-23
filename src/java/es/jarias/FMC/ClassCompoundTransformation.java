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

import java.util.ArrayList;
import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * @author jarias
 *
 */
public class ClassCompoundTransformation  {
	
    private Instances transformedFormat;

    /**
     * Returns the format of the transformed instances
     * 
     * @return the format of the transformed instances
     */
    public Instances getTransformedFormat() {
        return transformedFormat;
    }
    
    Instances data;
    int numLabels;
    int[] labelIndices;
    Attribute newClass;

    /**
     * 
     * @param mlData
     * @return the transformed instances
     * @throws Exception
     */
    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        data = mlData.getDataSet();
        numLabels = mlData.getNumLabels();
        labelIndices = mlData.getLabelIndices();

        Instances newData = null;

        // This must be different in order to combine ALL class states, not only existing ones.
        // gather distinct label combinations
        // ASSUME CLASSES ARE BINARY
        
        ArrayList<LabelSet> labelSets = new ArrayList<LabelSet>();
        
        double[] dblLabels = new double[numLabels];
        double nCombinations = Math.pow(2, numLabels);
        

        for (int i= 0; i < nCombinations; i++)
        {
            for (int l= 0; l < numLabels; l++)
            {
            	int digit = (int) Math.pow(2, numLabels-1-l);
            	dblLabels[l] = (digit & i) / digit;
            }
            
        	LabelSet labelSet = new LabelSet(dblLabels);
        	labelSets.add(labelSet);
        }
        
//        for (int i = 0; i < numInstances; i++) {
//            // construct labelset
//            double[] dblLabels = new double[numLabels];
//            for (int j = 0; j < numLabels; j++) {
//                int index = labelIndices[j];
//                dblLabels[j] = Double.parseDouble(data.attribute(index).value((int) data.instance(i).value(index)));
//            }
//            LabelSet labelSet = new LabelSet(dblLabels);
//
//            // add labelset if not already present
//            labelSets.add(labelSet);
//        }
        

        
        // create class attribute
        ArrayList<String> classValues = new ArrayList<String>(labelSets.size());
        for (LabelSet subset : labelSets) {
            classValues.add(subset.toBitString());
        }
        newClass = new Attribute("class", classValues);
        
//        for (String s : classValues)
//        {
//        	System.out.print(s+", ");
//        	
//        }
//        System.out.println();
        
        
        // remove all labels
        newData = RemoveAllLabels.transformInstances(data, labelIndices);

        // add new class attribute
        newData.insertAttributeAt(newClass, newData.numAttributes());
        newData.setClassIndex(newData.numAttributes() - 1);

        // add class values
        for (int i = 0; i < newData.numInstances(); i++) {
            //System.out.println(newData.instance(i).toString());
            String strClass = "";
            for (int j = 0; j < numLabels; j++) {
                int index = labelIndices[j];
                strClass = strClass + data.attribute(index).value((int) data.instance(i).value(index));
            }
            //System.out.println(strClass);
            newData.instance(i).setClassValue(strClass);
        }
        transformedFormat = new Instances(newData, 0);
        return newData;
    }

    /**
     * 
     * @param instance
     * @param labelIndices
     * @return tranformed instance
     * @throws Exception
     */
    public Instance transformInstance(Instance instance, int[] labelIndices) throws Exception {
        Instance transformedInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
        transformedInstance.setDataset(null);
        transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
        transformedInstance.setDataset(transformedFormat);
        return transformedInstance;
    }
    
    
}