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
import weka.core.Statistics;
import weka.core.Utils;
import weka.filters.Filter;

/**
 * 
 * @author jarias
 *
 */
public class BDeu {

	public final static int MAXP = 3;
	
	Instances dataset;
	
	int m_numAttributes;
	int m_numCases;
	int [] m_numValues;
	
	double [] singleScore;
	
	double [][] pairScores = new double[m_numAttributes][m_numAttributes];
	double [][] pairDiff = new double[m_numAttributes][m_numAttributes];
	
 	ArrayList<Integer> [] PC1;
 	HashSet<Integer>[] PC;
	
 	
	
	public BDeu(Instances data)
	{
		dataset = data;
		m_numAttributes = dataset.numAttributes();
		m_numCases = dataset.numInstances();
		
		m_numValues = new int[m_numAttributes];
		for (int att = 0; att < m_numAttributes; att++)
			m_numValues[att] = dataset.attribute(att).numValues();

		singleScore = new double[m_numAttributes];
		
		for (int i = 0; i < m_numAttributes; i++)
			singleScore[i] = localBdeuScore(i, new Integer[]{});
		
		
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
		pairScores = new double[m_numAttributes][m_numAttributes];
		pairDiff = new double[m_numAttributes][m_numAttributes];
		
		for (int i = 0; i < m_numAttributes; i++)
			for (int j = 0; j < m_numAttributes; j++)
			{
				if (i == j)
					continue;
				
				pairScores[i][j] = localBdeuScore(i, new Integer[]{j});
				pairDiff[i][j] = singleScore[i] - pairScores[i][j];
			}
		
		ArrayList<Integer> PC = new ArrayList<Integer>();
		
		ArrayList<Integer> open = new ArrayList<Integer>();
		for (int i = 0; i < m_numAttributes; i++)
			if (i != T)
				open.add(i);
		
		Collections.sort(open, new Comparator<Integer>() {

			public int compare(Integer o1, Integer o2) {
				
				if (pairDiff[T][o1] > pairDiff[T][o2])
					return 1;
				else if (pairDiff[T][o1] < pairDiff[T][o2])
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
					HashSet<Integer> parents = new HashSet<Integer>();
					parents.add(X);
					parents.addAll(Z);
					
					double diff = localBdeuScore(T, Z.toArray(new Integer[Z.size()])) - localBdeuScore(T, parents.toArray(new Integer[parents.size()]));
					
					if ( diff > 0)
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
					
					HashSet<Integer> parents = new HashSet<Integer>();
					parents.add(X);
					parents.addAll(S);

					double diff = localBdeuScore(T, S.toArray(new Integer[S.size()])) - localBdeuScore(T, parents.toArray(new Integer[parents.size()]));
					
					if ( diff > 0)
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
	
	protected double localBdeuScore(int X,  Integer[] parents) {
		
		int numParents = parents.length;

		// Hyper-Parameters
		double ess = 10;
		double kappa = 1 / (ess + 1);

		
		// Count number of values for each parent
		int cardinality = 1;
		for (int i : parents) {
			cardinality *= m_numValues[i];
		}

		int[][] Ni_jk = new int[cardinality][m_numValues[X]];
		double Np_ijk = (1.0 * ess) / (m_numValues[X] * cardinality);
		double Np_ij = (1.0 * ess) / cardinality;

		// initialize
		for (int j = 0; j < cardinality; j++)
			for (int k = 0; k < m_numValues[X]; k++)
				Ni_jk[j][k] = 0;

		for (int i = 0; i < m_numCases; i++) {
			int iCPT = 0;
			for (int iParent = 0; iParent < numParents; iParent++) {
				iCPT = (int) (iCPT * m_numValues[iParent]
						+ dataset.instance(i).value(parents[iParent]));
			}
			Ni_jk[iCPT][(int) dataset.instance(i).value(X)]++;
		}

		double fLogScore = 0.0;

		for (int iParent = 0; iParent < cardinality; iParent++) {
			double N_ij = 0;
			double N_ijk = 0;

			for (int iSymbol = 0; iSymbol < m_numValues[X]; iSymbol++) {
				if (Ni_jk[iParent][iSymbol] != 0) {
					N_ijk = Ni_jk[iParent][iSymbol];
					fLogScore += Statistics.lnGamma(N_ijk + Np_ijk);
					fLogScore -= Statistics.lnGamma(Np_ijk);
					N_ij += N_ijk;
				}
			}
			if (Np_ij != 0)
				fLogScore += Statistics.lnGamma(Np_ij);
			if (Np_ij + N_ij != 0)
				fLogScore -= Statistics.lnGamma(Np_ij + N_ij);
		}
		fLogScore += Math.log(kappa) * cardinality * (m_numValues[X] - 1);

		return fLogScore;
	}// localBdeuScore
	//--------------------------------------------------------------
}
