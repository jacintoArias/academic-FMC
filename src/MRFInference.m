%{

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

%}

function res=MRFClassify(nfolds, ipath, inference)

    % Add paths to data and libraries
    addpath(genpath('../lib/UGM'));
    addpath(genpath(ipath));

    disp(['Test Phase using ' inference ' inference:']);

    sumGlobal = 0;
    sumHamming = 0;
    % Main loop for cross validation
    for fold=0:nfolds-1,
    
        %%%%%%%%%%%%%%%%%%%%
        % Load generated data from 1st stage
        %
        conf = load(fullfile(ipath, ['conf_' num2str(fold) '.txt']));
        
        % Different dimensions in the distributions, they are loaded manually into a cell array
        distFile = fopen(fullfile(ipath, ['dist_' num2str(fold) '.txt']));
        distText = textscan(distFile,'%s', 'delimiter', '\n');
        
        % Problem with textscan, it returns a cell inside a cell array
        distText = distText{1};
        dist = cell(length(distText),1);
        
        % Extract probabilities from vectors
        for i=1:length(distText)
            temp = textscan(distText{i}, '%f', 'delimiter', '\t');
            dist{i} = temp{1};
        end
        
        % Load singleton
        sing = load(fullfile(ipath, ['sing_' num2str(fold) '.txt']));
        
        % Layout file outlines three different data: #states + #pairs + #singleton. That read manually
        layoFile = fopen(fullfile(ipath, ['layo_' num2str(fold) '.txt']));
        layo1 = str2num(fgets(layoFile));
        layo2 = str2num(fgets(layoFile));

        nStates = layo1;
        nSing  = layo2(2);
        nPairs = layo2(1);
        nCases = size(conf, 1);
        nNodes = size(conf, 2);

        distSize = nPairs;

        % Separate configurations and distributions for each instance
        for j=1:nCases,
            elm.instance = conf(j,:);   
            index = (distSize * (j-1))+1;
            elm.distributions = dist(index:index+distSize-1, :);
            MLData{j} = elm;
        end

        % Separate singleton predictions
        for j=1:nCases,
           index = (nSing * (j-1))+1;
           spred = sing(index:index+nSing-1,:);
           SPRED{j} = spred;
        end

        % Extract the correct classifications
        actualConf = zeros(nCases,nNodes);
        for instance=1:nCases-1,
            actualConf(instance,:) =  MLData{instance}.instance;
        end

        % Extract adjacencies from first distribution:
        tempStruct = MLData{1}.distributions;
        example = zeros(length(tempStruct),2);
        for i=1:length(tempStruct)
            example(i,:) = transpose(tempStruct{i}(1:2));
        end

        % Make adjacency matrix, it takes the actual pairs from the first distribution example:
        adj = zeros(nNodes,nNodes);
        for row=1:size(example, 1),
            adj(example(row,1)+1, example(row,2)+1) = 1;
            adj(example(row,2)+1, example(row,1)+1) = 1;
        end

        % Make edge structure
        edgeStruct = UGM_makeEdgeStruct(adj,nStates);
        maxState = max(edgeStruct.nStates);
        
        % Set up the nodes potentials (here we just specify uniform distributions)
        nodePot = ones(nNodes, maxState);

        % Data structure containing the class predictions
        predictions = zeros(nCases,nNodes);

        % Iterate over the instances to be classified. For each instances, we
        % build a pair-wise MRF with the class distributions as the pairwise edge
        % distributions. Afterwards, we find the most likely configuration as our
        % prediction

        for instance=1:nCases,

            maxState = max(edgeStruct.nStates);
            edgePot = zeros(maxState,maxState,edgeStruct.nEdges);

            % Set up the MRF for the current instance
            for e = 1:edgeStruct.nEdges
                edgeNodes = edgeStruct.edgeEnds(e,:);

                % Saves the distributions for this instance
                dists = MLData{instance}.distributions;

                % Retrieves the class pair indexes from the distributions
                pairIndexes = zeros(length(dists),2);
                for i=1:length(dists)
                    pairIndexes(i,:) = transpose(dists{i}(1:2));
                end

                % Find row for the corresponding edge (first it filters corresponding column and then select row)
                [r1,c1] = find((pairIndexes(:,1)+1) == edgeNodes(1));
                [r2,c2] = find((pairIndexes(r1,2)+1) == edgeNodes(2));
                
                % Extract distribution and build matrix out of it depending number of states
                potential = transpose(dists{r1(r2)});
                potential = potential(3:end);
                potential = vec2mat(potential, nStates(edgeNodes(2)));
                
                %Pad matrix with zeros to match maxstate
                padding = double([maxState-nStates(edgeNodes(1)) maxState-nStates(edgeNodes(2))]);
                potential = padarray(potential,padding, 'post');
                edgePot(:,:,e) = potential; 
            end
            
            % Perform MPE inference for global accuracy
            if strcmp(inference, 'exact')
                optimalDecoding = double(UGM_Decode_Exact(nodePot,edgePot,edgeStruct));
            else
                optimalDecoding = double(UGM_Decode_LBP(nodePot,edgePot,edgeStruct));
            end
            
            % We index states form 0 and not from 1, so subtract 1
            predictions(instance,:) = optimalDecoding-1';
            
            % Perform most probrable configuration for hamming acc
            if strcmp(inference, 'exact')
                [infRes,e,l] = UGM_Infer_Exact(nodePot,edgePot,edgeStruct);
                infResults{instance}=infRes;
            else
                [infRes,e,l] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);
                infResults{instance}=infRes;
            end
            
            % Exctract singleton predictions
            singlePred = SPRED{instance};
            for s=1:nSing,
               predictions(instance, singlePred(s, 2)+1) = singlePred(s, 3);
            end    
        end


        % Do the comparison
        
        % Hamming accuracy:
        correct = 0;
        for instance=1:nCases,
            [val,idx] = max(infResults{instance},[],2);
            predStates = idx'-1;
            
            singlePred = SPRED{instance};
            for s=1:nSing,
               predStates(singlePred(s, 2)+1) = singlePred(s, 3);
            end    
            
            correct = correct + sum(predStates==actualConf(instance,:));
        end
        marginal = correct/(nCases*nNodes); 
    
        % Global accuracy
        correct = 0;
        for instance=1:nCases,
             correct = correct + isequal(predictions(instance,:),actualConf(instance,:)); 
        end
        exact = correct/nCases;
    
        % Print
        fprintf('Fold %d: Global_acc: %.4f, Hamming_acc: %.4f\n',fold,exact,marginal);
        
        sumGlobal  = sumGlobal + exact;
        sumHamming = sumHamming + marginal;        
    end
    
    fprintf('\nMean Global_acc: %.4f\nMean Hamming_acc: %.4f\n\n', sumGlobal/nfolds, sumHamming/nfolds);
    
    
    
    