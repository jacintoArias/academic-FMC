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
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;

import mulan.multidimensional.data.MultiDimensionalInstances;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils;
import weka.filters.Filter;

/**
 * 
 * @author jarias
 *
 */
public class HITON {

	public final static int MAXP = 3;
	
	Instances dataset;
	
	int m_numAttributes;
	int m_numCases;
	int [] m_numValues;
	
	double [][] m_counts;
	double [][][] m_condiCounts;
	
	double [][] I;
	
	double [] m_probs;
	double [][] m_condiProbs;
	
 	ArrayList<Integer> [] PC1;
 	HashSet<Integer>[] PC;
	
	
	public HITON(Instances data)
	{
		dataset = data;
		m_numAttributes = dataset.numAttributes();
		m_numCases = dataset.numInstances();
		
		m_numValues = new int[m_numAttributes];
		for (int att = 0; att < m_numAttributes; att++)
			m_numValues[att] = dataset.attribute(att).numValues();
		
		m_counts = new double[m_numAttributes][];
		for (int att = 0; att < m_numAttributes; att++)
			m_counts[att] = new double[m_numValues[att]];
		
		m_condiCounts = new double[m_numAttributes][m_numAttributes][];
		for (int att1 = 0; att1 < m_numAttributes; att1++)
		{
			for (int att2 = att1+1; att2 < m_numAttributes; att2++)
			{
				m_condiCounts[att1][att2] = new double[m_numValues[att1]*m_numValues[att2]];
				m_condiCounts[att2][att1] = new double[m_numValues[att1]*m_numValues[att2]];
			}
		}		
		
		I = new double[m_numAttributes][m_numAttributes];
		
		
		// Compute counts:
		for (Instance inst : dataset)
		{
			for (int att1 = 0; att1 < m_numAttributes; att1++)
			{
				m_counts[att1][(int) inst.value(att1)]++;
				
				for (int att2 = att1+1; att2 < m_numAttributes; att2++)
				{
					m_condiCounts[att1][att2][(int) (inst.value(att1) * m_numValues[att2] + inst.value(att2))]++;
					m_condiCounts[att2][att1][(int) (inst.value(att2) * m_numValues[att1] + inst.value(att1))]++;
				}
			}
		}
		
		
		// Compute I(X_i; X_j)
		for (int i = 0; i < m_numAttributes; i++)
		{
			for (int j = 0; j < m_numAttributes; j++)
			{
				if (i == j)
					continue;
				
				double mi = 0.0;
				for (int v_i = 0; v_i < m_numValues[i]; v_i++)
				{
					for (int v_j = 0; v_j < m_numValues[j]; v_j++)
					{
						int condiIndex = (int) (v_i * m_numValues[j] + v_j);
						if ((1.0 * m_numCases * m_condiCounts[i][j][condiIndex]) / (1.0 * m_counts[i][v_i] * m_counts[j][v_j]) > 0)
							mi += m_condiCounts[i][j][condiIndex] * Math.log((1.0 * m_numCases * m_condiCounts[i][j][condiIndex]) / (1.0 * m_counts[i][v_i] * m_counts[j][v_j]));
					}
				}
				I[i][j] = mi / m_numCases;
			}
		}
		
		// Compute parent and children for every variable:
		PC1 = new ArrayList[m_numAttributes];
		for (int i = 0; i < m_numAttributes; i++)
			PC1[i] = HITONPC1(i);
		
		PC = new HashSet[m_numAttributes];
		for (int i = 0; i < m_numAttributes; i++)
			PC[i] = HITONPC2(i);
		
	}
	
	private HashSet<Integer> HITONPC2(final int T)
	{
		HashSet<Integer> PC = new HashSet<Integer>();
		
		for (int X : PC1[T])
		{
			if (PC1[X].contains(T))
				PC.add(X);
		}
		
		return PC;
		
	}
	
	private ArrayList<Integer> HITONPC1(final int T)
	{
		ArrayList<Integer> PC = new ArrayList<Integer>();
		
		ArrayList<Integer> open = new ArrayList<Integer>();
		for (int i = 0; i < m_numAttributes; i++)
			if (i != T && I[T][i] != 0)
				open.add(i);
		
		Collections.sort(open, new Comparator<Integer>() {

			public int compare(Integer o1, Integer o2) {
				
				if (I[T][o1] < I[T][o2])
					return 1;
				else if (I[T][o1] > I[T][o2])
					return -1;
				else
					return 0;
			}
		});
		
		
		
		while (open.size() > 0)
		{
//			System.out.println("OPEN: "+open.toString());	
			PC.add(open.remove(0));
//			System.out.println("REMOVES: "+PC.get(PC.size()-1) +"  PC: "+PC.toString());
			
			for (int v_pc = 0; v_pc < PC.size(); v_pc++)
			{
				int X = PC.get(v_pc);
				
				boolean keep = true;
//				System.out.println("\tFOR: "+X);
				
				ArrayList<ArrayList<Integer>> combinations = combine(PC, X, MAXP);
				
				for (ArrayList<Integer> Z: combinations)
				{
//					System.out.print("\t"+Z.toString()+"\t");
					double p = chisq(computeConditionalMI(X, T, Z.toArray(new Integer[Z.size()])), X, T);
					
					if ( p >= 0.05)
					{
//						System.out.print(X+" is rejected");
//						System.out.println(" p = "+p+" ");
						keep = false;
						break;
					}
//					System.out.println(" p = "+p+" ");
				}

				if (!keep)
					PC.remove(v_pc);
				
//				System.out.println();
			}
			
//			System.out.println();
//			System.out.println("  PC: "+PC.toString());
		}
		
		
		return PC;
	}
	
	public HashSet<Integer> HITONMB(int T)
	{
		HashSet<Integer> V = new HashSet<Integer>();
		for (int i = 0; i < m_numAttributes; i++)
			if (i != T)
				V.add(i);
		
		HashSet<Integer> MB = new HashSet<Integer>();
		for (int i : PC[T])
		{
			MB.add(i);
		}
		
		HashSet<Integer> PCPC = new HashSet<Integer>(); 
		for (int i : MB)
		{
			for (int j : PC[i])
			{
				PCPC.add(j);
			}
		}
		MB.addAll(PCPC);
		MB.remove(T);
		
		HashSet<Integer> MBCopy = new HashSet<Integer>();
		MBCopy.addAll(MB);
		
		// Find spouses:
		for (int X : MBCopy)
		{
			for (int Y : PC[T])
			{
				if (!PC[Y].contains(X))
					continue;
				
				V.remove(X);
				V.remove(Y);
				ArrayList<Integer> combs = new ArrayList<Integer>();
				combs.addAll(V);
				V.add(X);
				V.add(Y);
				
				ArrayList<ArrayList<Integer>> combinations  = combine(combs, -1, MAXP);
				
				for (ArrayList<Integer> S : combinations)
				{
					S.add(Y);
					double p = chisq(computeConditionalMI(X, T, S.toArray(new Integer[S.size()])), X, T);
					
					if ( p >= 0.05)
					{
						MB.remove(X);
						break;
					}
				}
			}
		}
		
		return MB;
	}
	
	private ArrayList<ArrayList<Integer>> combine(ArrayList<Integer> list, int exclude, int maxP) 
	{
		ArrayList<ArrayList<Integer>> combinations = new ArrayList<ArrayList<Integer>>();
		
		
		for (int v_0 = 0; v_0 < list.size(); v_0++)
		{
			int e_0 = list.get(v_0);
			
			if (e_0 == exclude)
				continue;
			
			ArrayList<Integer> C = new ArrayList<Integer>();
			
			C.add(e_0);
			combinations.add(C);
			
			for (int v_1 = v_0+1; v_1 < list.size(); v_1++)
			{
				int e_1 = list.get(v_1);
				
				if (C.size() == maxP)
					break;
				
				if (e_1 == exclude)
					continue;
				
				C.add(e_1);
				combinations.add(C);
			}
			
		}

		return combinations;
	}
	
	public double computeConditionalMI(int X, int Y, Integer[] Z)
	{
		
		int Zcardinality = 1;
		int [] offsets = new int[Z.length]; 
			
		for (int z = Z.length-1; z >= 0; z--)
		{
			offsets[z] = Zcardinality;
			Zcardinality *= m_numValues[Z[z]];
		}
		offsets[Z.length-1] = 0;
		
		double conditional2Counts[][][] = new double[m_numValues[X]][m_numValues[Y]][Zcardinality];
		double conditionalXCounts[][] = new double[m_numValues[X]][Zcardinality];
		double conditionalYCounts[][] = new double[m_numValues[Y]][Zcardinality];
		
		double Zcounts[] = new double[Zcardinality];
		
		for (Instance inst : dataset)
		{
			int index = 0;
			for (int z = 0; z < Z.length; z++)
				index += inst.value(Z[z]) * offsets[z];
			index += inst.value(Z[Z.length-1]);
			
			Zcounts[index]++;
			conditionalXCounts[(int) inst.value(X)][index]++;
			conditionalYCounts[(int) inst.value(Y)][index]++;
			conditional2Counts[(int) inst.value(X)][(int) inst.value(Y)][index]++;
		}
		
		// COMPUTE I(X;Y|Z)
		double mi = 0.0;
		for (int v_z = 0; v_z < Zcardinality; v_z++)
		{
			double p_z = Zcounts[v_z] / m_numCases;
			double t_mi = 0.0;
			for (int v_x = 0; v_x < m_numValues[X]; v_x++)
			{
				for (int v_y = 0; v_y < m_numValues[Y]; v_y++)
				{
					if ((1.0 * conditional2Counts[v_x][v_y][v_z] / Zcounts[v_z]) / ( (conditionalXCounts[v_x][v_z] / Zcounts[v_z]) * (conditionalYCounts[v_y][v_z] / Zcounts[v_z]) ) > 0)
					{
						t_mi += (conditional2Counts[v_x][v_y][v_z]/Zcounts[v_z]) * 
						Math.log( (1.0 * conditional2Counts[v_x][v_y][v_z] / Zcounts[v_z]) / 
								( (conditionalXCounts[v_x][v_z] / Zcounts[v_z]) * (conditionalYCounts[v_y][v_z] / Zcounts[v_z]) ) );
					}
				}
			}
			mi += p_z * t_mi;
		}
		
		return mi;
	}
	
	public double chisq(double mutualInfo, int x, int y)
	{
		
		int degreesOfFreedom;
		double dxyz,chiSquare;
	    
	    dxyz = mutualInfo;
	    chiSquare = 2.0 * m_numCases * dxyz;

	    degreesOfFreedom = (m_numValues[x] - 1) * (m_numValues[y] - 1);

	    if(degreesOfFreedom <= 0) degreesOfFreedom = 1;

	    double p;
	    if (chiSquare > 0)
	    	p = 1-statistics.StatFunctions.pchisq(chiSquare, degreesOfFreedom);
	    else
	    	p = 0;
	    
	    return p;
	}
	
}
