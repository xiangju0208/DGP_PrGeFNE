function [P_G, P_D, Pt ] = A_DGP_PrGeFNE_all(AdjGfG,AdjGfD,AdjDfD, ... 
    AdjGfP,  AdjDfP, AdjPfP,  ... 
    AdjGfGO, AdjDfGO,AdjPfGO, AdjGOfGO, ...  
    P0_G,P0_D,  paraReNet, paraProp, isdebug) 
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    if nargin<1 || isempty (AdjGfG)
        AdjGfP=[],  AdjDfP=[], AdjPfP=[],  ... 
        AdjGfGO=[], AdjDfGO=[],AdjPfGO=[], AdjGOfGO=[], ... 
    %     N_gene=300; N_disease = 10; 
        N_gene=100; N_disease = 100; 
        AdjGfG = sparse( rand(N_gene,N_gene)>0.2 ); 
        AdjGfD = sparse( rand(N_gene,N_disease)>0.2 ); 
        AdjDfD = sparse( rand(N_disease,N_disease)>0.2 ); 
        P0_G = zeros(N_gene,1); P0_G(1:3)=1; P0_G = P0_G./sum(P0_G);
        P0_D = zeros(N_disease,1); P0_D(1:2)=1; P0_D = P0_D./sum(P0_D);
        %
        tic  
        disp('start.........') 
        %
        idx = find( AdjGfD ); n_pos = length( idx); 
        ind_fold = crossvalind('Kfold', n_pos, 2) ; 
        AdjGfD1 = AdjGfD; AdjGfD1(idx(ind_fold~=1) ) =0; 
        AdjGfD2 = AdjGfD; AdjGfD2(idx(ind_fold==1) ) =0; 
        AdjGfD  = AdjGfD2; % used to train the model.  
        DisIDset = 1: size( AdjDfD,1 ); 
        DisIDset = [1:10]; 
        P0_G = AdjGfD(:,  DisIDset   );   
        P0_D = speye( size( AdjDfD) );  P0_D=P0_D(:,DisIDset); 

        restart = 0.7;   pro_jump = 0.5;  eta =0.5;   
        warning('TestTestTestTestTestTestTestTestTestTestTestTestTestTestTest'); 
        isdebug =true; 
        istest  =true; 
        NormalizationType = 'ProbabilityNormalizationColumn';      
    end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    % Input % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % AdjGfG: associatins from (f) genes (G) to Genes (G)    
    % AdjGfD: associatins from Diseases (D) to Genes (G) GfD
    % AdjDfD  associatins from Diseases (D) to Disease (G) 
    % AdjGfP     Genes (G) from (f) Phenotype (P)   
    % AdjDfP     Diseases (D) from (f) Phenotype (P)    
    % AdjPfP     Phenotype (P) from (f) Phenotype (P)    
    % AdjGfGO    genes (G) from (f) GO terms (GO)   
    % AdjDfGO    Diseases (D) from (f) GO terms (GO)   
    % AdjPfGO    Phenotype (P) from (f) GO terms (GO)   
    % AdjGOfGO   GO terms (GO) from (f) GO terms (GO)     
    % 
    % P0_G: column vector (set) initial probabilities in Gene network
    % P0_D: column vector (set) initial probabilities in Disease network
    % P0_G and P0_D must have the same # of columns. 
    % gamma/restart: restarting Probability  
    % pro_jump: jumping Probability between different networks
    % eta: ratio of Probability in second network
    % NormalizationType = 'ProbabilityNormalizationColumn'; %%for 'Random Walk' RWR, RWRH  RWRM  RWRMH and more   
    % NormalizationType = 'LaplacianNormalization'; %%  for label propagation, prince and more....    
    % Ouput % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % P_G: stable probabilities in Gene network 
    % P_D: stable probabilities in Disease network  
    % Pt: stable probabilities in Gene+disease network 
    % % % % % % % % % % % % % % % % % % % % % %  
    % Reference: 
    % Xiang, et al, PrGeFNE: Predicting disease-related genes by fast network embedding, Methods, 2020,https://doi.org/10.1016/j.ymeth.2020.06.015. 
    % Zhang, et al, Billion-scale network embedding with iterative random projection. In 2018 IEEE International Conference on Data Mining (ICDM) (pp. 787-796). IEEE.
    % By Ju Xiang, 
    % Email: xiang.ju@foxmail.com, xiangju@csu.edu.cn  
    % 2021-4-8 
    %    
    % % % % % % % % % % % % % % % % % 
    % % % Example  
    % % % paraReNet.knn_gene  = 100; 
    % % % paraReNet.knn_dis   = 20  ;  
    % % % % % % % % %  
    % % % paraProp.restart  = 0.7 ; 
    % % % paraProp.pro_jump = 0.8 ;
    % % % paraProp.eta      = 0.5 ;
    % % % paraProp.NormalizationType = 'ProbabilityNormalizationColumn';  
    % % %
    if ~exist('paraReNet','var') || isempty( paraReNet )
        knn_gene  = 100 ; 
        knn_dis   = 20  ;  
    else
        knn_gene  = paraReNet.knn_gene; 
        knn_dis   = paraReNet.knn_dis; 
    end
    % % % 
    if ~exist('paraProp','var') || isempty( paraProp )
        restart  = 0.7 ; 
        pro_jump = 0.8 ;
        eta      = 0.5 ;
        NormalizationType = 'ProbabilityNormalizationColumn';  
    else
        restart  = paraProp.restart; 
        pro_jump = paraProp.pro_jump;
        eta      = paraProp.eta;
        NormalizationType = paraProp.NormalizationType;
    end
    % % % % % % % % % % % % % % %     
    if ~exist('pro_jump','var') || isempty (pro_jump)
        pro_jump = 0.5; 
    end   
    if ~exist('eta','var') || isempty (eta)
        eta = 0.5; 
    end   
	
    if ~exist('NormalizationType','var') || isempty (NormalizationType)
        NormalizationType = 'ProbabilityNormalizationColumn' ; 
    elseif ~ismember(NormalizationType,{ 'LaplacianNormalization', 'column','col',  'ProbabilityNormalizationColumn','ProbabilityNormalizationCol'     } )
        error(['NormalizationType is wrong: ',char(string(NormalizationType)) ]);
    end   
	
    if ~exist('isdebug','var') || isempty (isdebug)
        isdebug = false;  
    end   
    %  
    if isempty( AdjDfD )
       AdjDfD = speye(N_disease);  
       warning('AdjDfD is empty.');
    end
    % 
    [N_gene, N_disease] = size( AdjGfD );
    if size(P0_G,1)~=N_gene; error( 'P0_G must be column vector(set), with length of the number of genes.'  );end
    if size(P0_D,1)~=N_disease; error( 'P0_D must be column vector(set), with length of the number of diseases.'  );end
    % 
    % 
    Aunion = sparse( ...
             [AdjGfG,  AdjGfD,  AdjGfP,  AdjGfGO; ...   
              AdjGfD', AdjDfD,  AdjDfP,  AdjDfGO; ... 
              AdjGfP', AdjDfP', AdjPfP,  AdjPfGO; ... 
              AdjGfGO',AdjDfGO',AdjPfGO',AdjGOfGO  ]  );    
    ind_gene = [1: N_gene];
    ind_dis  = [N_gene+1: N_gene+N_disease];
	
    % % % % % % % % % % % % % % % % % % % % % % %      
    paras.d       = 128 ;
    paras.Ortho   = 1  ;
    % embedding for adjacency matrix for reconstruction
    % %             paras.q       = 2 ;
    % %             paras.weights = [1,0.1,0.001];  
    % embedding for transition matrix for classification
    % paras.q       = 3 ;
    % paras.weights = [1,1e2,1e4,1e5];   
    paras.seed    = 0;
    %     savefilename = filestr.embtxt; 
    worktype = 'classification';  
    % %             worktype = 'reconstruction'; 
    features = getRandNEemb_in(Aunion,paras, [], [], worktype);    
    Aunion = []; 
	%
    features_gene = features(ind_gene,2:end);  
    features_dis  = features(ind_dis,2:end);  
    features =[]; 
    %
    features_gene = features_gene./(max( abs( features_gene ), [] , 2) +eps ) ; 
    features_gene = features_gene./sqrt( sum( features_gene.^2 , 2) +eps ) ;   
    SimMatrix = features_gene*features_gene';  
    %
    symmetrized   = true ; keepweight    = true ; 
    AdjGfG_rc = sparse( getAdjKnnColumns_in( SimMatrix,  knn_gene , symmetrized, keepweight ) );  SimMatrix =[];     
    
	%
    features_dis = features_dis./(max( abs( features_dis ), [] , 2) +eps ) ; 
    features_dis = features_dis./sqrt( sum( features_dis.^2 , 2) +eps ) ;   
    SimMatrix = features_dis*features_dis';      
    AdjDfD_rc = sparse( getAdjKnnColumns_in( SimMatrix,  knn_dis , symmetrized, keepweight ) );  SimMatrix =[];      
    %  
	%  
    [ M_rc , IsNormalized ] = getNormalizedMatrix_Heter(AdjGfG_rc,AdjGfD,AdjDfD_rc, pro_jump,  NormalizationType, []) ; 
	% 
    if any( strcmpi(NormalizationType,{'col','ProbabilityNormalizationColumn','NormalizationColumn', 'Column'}) )
        P0_G = P0_G./(sum(P0_G,1)+eps);  
        P0_D = P0_D./(sum(P0_D,1)+eps); 
        P0 = [ (1-eta)*P0_G; eta*P0_D]; 
		P0 = P0./( sum(P0,1)+eps )  ;  
    else
        P0 = [ (1-eta)*P0_G; eta*P0_D];
    end    
    % [Pt, WAdj ]= A_RWRplus(Adj, r_restart, P0, N_max_iter, Eps_min_change, IsNormalized,  NormalizationType)
    Pt = A_RWRplus(M_rc, restart, P0 , [],[], IsNormalized);   
    P_G = Pt(1:N_gene,:);
    P_D = Pt(N_gene+1:end,:); 
	% 
	% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
    if isdebug
%         sum(M)  
%         disp('*******************************************************')
%         allP=sum(Pt) 
%         ss = sum(M_rc(:)) 
%         size( M_rc )  
        P_G=[]; P_D=[];Pt=[] ; 
    end
end
 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
function Adjknn = getAdjKnnColumns_in( SimMatrix,  k_neighbors_vec , symmetrized, keepweight )
% Input: 
% SimMatrix  similarity matrix
% k_neighbors  vector with number of neighbors of each node   
% symmetrized  1 or 0
% keepweight   keep similarity as weight of edges
% Output: Adjknn   matrix 
% Ju Xiang 
% 2019-5
    if isempty(SimMatrix) 
        % sort_dim = 2;
        SimMatrix =rand(10);   % for testing only 
        warning('test test test ');
    end 
    SZ =size( SimMatrix); 
    % 
    if isscalar(k_neighbors_vec)
        k_neighbors_vec = k_neighbors_vec(1)*ones( SZ(1), 1 ); 
    end
    if isempty(symmetrized) 
        symmetrized = true;
    end
    if isempty(keepweight) 
        keepweight = false;
    end
     
    if any( k_neighbors_vec>SZ(1)-1 )
        k_neighbors_vec(  k_neighbors_vec>SZ(1)   ) = SZ(1)-1;
        warning( ['There is k_neighbors:','>', num2str(SZ(1)-1),' the maximal number of neighbors'] );
    end
    
	% %     
    SimMatrix(   sub2ind( SZ, 1:SZ(1),1:SZ(1) )      ) = -inf;   %   
    % SimMatrix(   ( eye( SZ ) )==1       ) = -inf; 
    [~,II] = sort( SimMatrix ,2, 'descend' );  
    Adjknn = zeros( SZ );
    for ii=1: SZ(1)
        knn = II(ii,1: k_neighbors_vec(ii) ); 
        if keepweight
            Adjknn(ii,  knn ) = SimMatrix(ii,  knn );    
        else
            Adjknn(ii,  knn ) = 1; 
        end
    end 
    Adjknn(sub2ind( SZ, 1:SZ(1),1:SZ(1) )) = 0; 
    if symmetrized
        [i,j,v] = find( Adjknn ); 
        ind = sub2ind( SZ, i ,j );
        Adjknn = Adjknn' ; 
        Adjknn(ind) = v; 
        % %     Adjknn = Adjknn';
        % %     Adjknn(ind) = 
        % Adjknn = full
    end 
end

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
function [emb]=getRandNEemb_in(A,paras, savefilename,TableNode, worktype)
    % A sample run on the BlogCatalog Dataset
    % load data
    % load('BlogCatalog');
    if ~exist('A','var') 
        warning('test');
        nn =1000; 
        A= rand(nn); A = A + A';  savefilename = 't11111111111.emb.txt'; 
        paras.d       =128;
        paras.Ortho   = 1;
        paras.q       = 2 ;
        paras.weights = [1,0.1,0.001];  
        paras.seed    = 0;
        TableNode = table('ID'+string(1:nn)');
        TableNode =[]
    end
    % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    d       = paras.d;
    Ortho   = paras.Ortho;
    seed    = paras.seed;
    % % q       = paras.q;
    % % weights = paras.weights; 
    %
    N = length(A);
    % Common parameters
    % d = 128;
    % Ortho = 1;
    % seed = 0;
    switch worktype
        % worktype = 'classification';  'reconstruction';  
        case 'reconstruction'
            % embedding for adjacency matrix for reconstruction
            q = 2;
            weights = [1,0.1,0.001];
            U_list = RandNE_Projection(A,q,d,Ortho,seed);
            U = RandNE_Combine(U_list,weights);
            % prec = Precision_Np(A,sparse(N,N),U,U,1e6);
            % semilogx(1:1e6,prec);
        case 'classification'
            % % embedding for transition matrix for classification
            q = 3;
            weights = [1,1e2,1e4,1e5];
            A_tran = spdiags(1 ./ sum(A,2),0,N,N) * A;
            U_list = RandNE_Projection(A_tran,q,d,Ortho,seed);
            U = RandNE_Combine(U_list,weights);
            % % normalizing
            U = spdiags(1 ./ sqrt(sum(U .* U,2)),0,N,N) * U;
            % % Some Classification method, such as SVM in http://leitang.net/social_dimension.html
        otherwise
            error('No defintion')
    end
    [n_node, n_feature] = size( U ); 
    % Delimiter is space
    if ~isempty(TableNode)
        IDstr = TableNode{:,1}; 
    % % %     tbl_emb = table(IDstr, U);
    % % %     writetable(tbl_emb, savefilename, 'Delimiter', ' ' ,'WriteVariableNames',false );
        if ~isempty( savefilename )
            fileID = fopen(savefilename,'w+');
            fprintf(fileID,'%d %d\n', n_node, n_feature ); 
            fmt =strjoin(['%s',repmat({'%f'},1, n_feature),'\n'], ' ' ); 
            for ii=1:n_node
                fprintf(fileID,fmt, IDstr{ii}, U(ii,:)); 
            end
            fclose(fileID);
        end

    else
        U1=(0:N-1)';
        emb=[U1 U];
        if ~isempty( savefilename )
            fileID = fopen(savefilename,'w+');
            fprintf(fileID,'%d %d\n', n_node, n_feature ); 
            fmt =strjoin(['%d',repmat({'%f'},1, n_feature),'\n'], ' ' );   
            fprintf(fileID,fmt, emb'); 
            fclose(fileID);
        end
    end
end
% % % % % % % % % % % % % % % % % % % % % % % % % 
function U_list = RandNE_Projection(A,q,d,Ortho,seed)
    % Inputs:
    %   A: sparse adjacency matrix
    %   q: order
    %   d: dimensionality
    %   Ortho: whether use orthogonal projection
    %   seed: random seed
    % Outputs:
    %   U_list: a list of R, A * R, A^2 * R ... A^q * R

    N = size(A,1);

    rng(seed);                               % set random seed
    U_list = cell(q + 1,1);                       % store each decomposed part
    U_list{1} = normrnd(0,1/sqrt(d),N,d);         % Gaussian random matrix
    if Ortho == 1                            % whether use orthogonal projection
        U_list{1} = GS(U_list{1});
    end
    for i = 2: (q + 1)                       % iterative random projection
        U_list{i} = A * U_list{i-1};
    end
end


% % % % % % % % % % % % % % % % % % % % % % % % % 
function P_ortho = GS(P)
    % Input:
    %   P: n x d random matrix
    % Output:
    %   P_ortho: each column orthogonal while maintaining length
    % Performing modified Gram?CSchmidt process

    [~,d] = size(P);
    temp_l = zeros(d,1);
    for i = 1:d
        temp_l(i) = sqrt( sum(P(:,i) .^2) );
    end
    for i = 1:d
        temp_row = P(:,i);
        for j = 1:i-1
            temp_j =  P(:,j);
            temp_product = temp_j' * temp_row  / temp_l(j)^2;
            temp_row = temp_row - temp_product * temp_j ; 
        end
        temp_row = temp_row * (temp_l(i) / sqrt(temp_row' * temp_row));
        P(:,i) = temp_row;
    end
    P_ortho = P;
end

% % % % % % % % % % % % % % % % % % % 
function U = RandNE_Combine(U_list,weights)
    % Inputs:
    %   U_list: a list of decomposed parts, generated by RandNE_Projection
    %   weights: a vector of weights for each order, w_0 ... w_q
    % Outputs:
    %   U: final embedding vector

    if size(U_list,1) < length(weights)
        error('Weights not consistent');
    end
    U = weights(1) * U_list{1};
    for i = 2:length(weights)
        U = U + (weights(i) * U_list{i});
    end

end

% % % % % % % % % % % % % % % % % % % % 
function result = Precision_Np(Matrix_test,Matrix_train,U,V,Np)
    % Matrix_test is n x n testing matrix, may overlap with Matrix_train
    % Matrix_train is n x n training matrix
    % U/V are content/context embedding vectors
    % Np: returns Precision@Np for pairwise similarity 
    [N,~] = size(U);
    if (N > 30000)
        error('Network too large. Sample suggested.');
    else
        Sim = U * V';
        [temp_row,temp_col,temp_value] = find(Sim);
        clear Sim;
    end
    temp_choose = (Matrix_train(sub2ind([N,N],temp_row,temp_col)) == 0) & (temp_row ~= temp_col);
    temp_row = temp_row(temp_choose);
    temp_col = temp_col(temp_choose);
    temp_value = temp_value(temp_choose);
    clear temp_choose;
    [~,temp_index] = sort(temp_value,'descend');
    if length(temp_index) < Np
        error('Np too large');
    end
    temp_index = temp_index(1:Np);
    clear temp_value;
    temp_row = temp_row(temp_index);
    temp_col = temp_col(temp_index);
    clear temp_index;
    result = Matrix_test(sub2ind([N,N],temp_row,temp_col)) > 0;
    result = cumsum(result > 0) ./ (1:length(result))';
end

% % % % % % % % % % % 
% % % % % % % % % 
function WAdj = getNormalizedMatrix(Adj, NormalizationType, SetIsolatedNodeSelfLoop )
    if ~exist('Adj','var') 
        Adj =rand(5); dim=1;SetIsolatedNodeSelfLoop = true;  
        NormalizationType = 'col' ;
        % NormalizationType = 'laplacian normalization' ;
        istest = 1; 
        warning('Test Test Test Test Test Test Test ');
    end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     
% Adj  adjecent matrix
% % NormalizationType: 
% % 'probability normalization' for 'Random Walk' RWR, RWRH  RWRM  RWRMH and more   
% % 'laplacian normalization' for prince and more....
% SetIsolatedNodeSelfLoop    set isolated node
% >= Matlab 2016
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
%     if ~issparse(Adj)
%         Adj = sparse( Adj );
%     end   
    if ischar(NormalizationType)
    %         NormalizationType =  (NormalizationType);
        switch  lower( NormalizationType )
            case lower( { 'column','col',  ...
                    'ProbabilityNormalizationColumn','ProbabilityNormalizationCol',...
                    'ProbabilityColumnNormalization','ProbabilityColNormalization',...
                    'NormalizationColumn','NormalizationCol' , ...
                    'ColumnNormalization','ColNormalization'   })
                NormalizationName = 'ProbabilityNormalization' ;  %  'Random Walk'  
                dim =1;
            case lower({ 'row' ,'ProbabilityNormalizationRow' ,'NormalizationRow' ,'ProbabilityRowNormalization' ,'RowNormalization'   })
                NormalizationName = 'ProbabilityNormalization' ;  %  'Random Walk'  
                dim =2;
            case lower('LaplacianNormalization')
                NormalizationName = NormalizationType; 
            case lower('LaplacianNormalizationMeanDegree')
                NormalizationName = NormalizationType; 
            case lower('ColNorm2')
                NormalizationName = NormalizationType; 
            case lower('RowNorm2')
                NormalizationName = NormalizationType; 
            case lower({'none', 'None', 'NONE'})
                % NormalizationName = 'None'; 
                WAdj = Adj; 
                return; 
            otherwise
                error(['There is no type of normalization: ',char( string(NormalizationType) )] );
        end
        
    elseif isnumeric(  NormalizationType   ) 
        NormalizationName =  ( 'ProbabilityNormalization' ) ;  %  'Random Walk'  
        dim = NormalizationType; 
        
    elseif isempty( NormalizationType )
        WAdj = Adj; 
        return;  
        
    else; error('There is no defintion of NormalizationType')
    end 
    % NormalizationName = lower( NormalizationName );
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     %
    switch lower( NormalizationName )
        case lower( 'ProbabilityNormalization' )
            degrees = sum(Adj,dim);
            if any( degrees~=1)
                WAdj = Adj./ ( degrees+eps  );           
                % % WAdj = Adj./ repmat( degrees +eps,[size(Adj,1),1]); 
            else
                WAdj = Adj; 
            end
            % 
            if SetIsolatedNodeSelfLoop  && size(Adj,1)==size(Adj,2) 
                ii = find( ~degrees ); 
                idx = sub2ind( size(Adj), ii,ii ); 
                WAdj(idx) = 1;  % set to be 1 for isolated nodes, 
            end
            
        case lower( 'LaplacianNormalization')
            deg_rowvec = ( sum(Adj,1) ).^0.5;  
            deg_colvec = ( sum(Adj,2) ).^0.5;   
            WAdj = (Adj./(deg_colvec+eps))./(deg_rowvec+eps) ;    
            % 
            if SetIsolatedNodeSelfLoop && size(Adj,1)==size(Adj,2)
                ii = find( ~sum(Adj,2) ) ; 
                % size(  WAdj )
                % size(  Adj )
                WAdj( sub2ind( size(Adj), ii,ii ) ) = 1;  % set to be 1 for isolated nodes, 
            end
            
        case lower( 'LaplacianNormalizationMeanDegree')
            n_node = length( Adj );
            km = sum( Adj(:) )./ n_node;  
            WAdj = Adj./( (km.^0.5)*(km.^0.5)  +eps) ;    
            % 
            if SetIsolatedNodeSelfLoop  && size(Adj,1)==size(Adj,2)
                ii = find( ~sum(Adj,2) ); 
                WAdj( sub2ind( size(Adj), ii,ii ) ) = 1;  % set to be 1 for isolated nodes, 
            end
            
        case lower( {'ColNorm2'} )   
            WAdj = Adj./ ( sqrt(sum( Adj.^2 ,1 )) +eps ); 
            
        case lower( {'RowNorm2'} )    
            WAdj = Adj./ ( sqrt(sum( Adj.^2 ,2 )) +eps ); 
            
        case lower( {'None','none'} )
            WAdj = Adj;   % 不做任何处理  
        otherwise
            error(['NormalizationName is wrong: ',char(string(NormalizationName) )   ]);
    end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    if exist('istest','var') && istest 
        WAdj(1:5,1:5)
        sum(WAdj,1) 
        sum(WAdj,2)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ combMatrix , IsNormalized ] = getNormalizedMatrix_Heter(AdjGfG,AdjGfD,AdjDfD, pro_jump,  NormalizationType, isdebug) 
% % % % % % % % % % % % % % % % % % % % 
if ~exist('AdjGfG','var')|| isempty (AdjGfG)
    N_gene=30;
    N_disease = 10; 
    AdjGfG = rand(N_gene,N_gene)>0.2 ; 
    AdjGfD = rand(N_gene,N_disease)>0.2 ; 
    AdjDfD = rand(N_disease,N_disease)>0.2 ; 
    P0_G = zeros(N_gene,1); P0_G(1:3)=1; P0_G = P0_G./sum(P0_G);
    P0_D = zeros(N_disease,1); P0_D(1:2)=1; P0_D = P0_D./sum(P0_D);
    % 
    pro_jump = 0.5;  
    warning('TestTestTestTestTestTestTestTestTestTestTestTestTestTestTest'); 
    isdebug =true; 
    istest =true; 
    NormalizationType = 'ProbabilityNormalizationColumn'; %%for 'Random Walk' RWR, RWRH  RWRM  RWRMH and more   
%     NormalizationType = 'ProbabilityNormalizationRow'; %%for label propagation    
    % NormalizationType = 'LaplacianNormalization'; %%  for label propagation, prince and more....    
%     NormalizationType = 'Weight'; %% Weighting  ....    
%     NormalizationType = 'None'; %%  without normalization ....    
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Input % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% AdjGfG: associatins from (f) genes (G) to Genes (G)  
% AdjGfD: associatins from Diseases (D) to Genes (G) GfD
% AdjDfD  associatins from Diseases (D) to Disease (G)  
% pro_jump: jumping Probability from first layer to second layer or weighting the effect of second layer on the first layer.    
% NormalizationType = 'ProbabilityNormalizationColumn'; %%for 'Random Walk' RWR, RWRH  RWRM  RWRMH and more   
% NormalizationType = 'ProbabilityNormalizationRow'; %%for label propagation    
% NormalizationType = 'LaplacianNormalization'; %%  for label propagation, prince and more....    
% NormalizationType = 'Weight'; %% Weighting  ....    
% NormalizationType = 'None'; %%  without normalization ....    
% Ouput % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% combMatrix is matrix after normalization.   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% By Ju Xiang, 
% Email: xiang.ju@foxmail.com, xiangju@csu.edu.cn  
% 2019-8-2 
    % global   Global_Var_RWRH 
    if ~exist('pro_jump','var') || isempty (pro_jump)
        pro_jump = 0.5; 
    elseif pro_jump>1 || pro_jump <0
        error('pro_jump is wrong. it should be between 0 and 1');
    end      
% %     if ~exist('NormalizationType','var') || isempty (NormalizationType)
% %         NormalizationType = 'ProbabilityNormalizationColumn' ; 
% %     elseif ~ismember(NormalizationType,{ 'LaplacianNormalization', 'column','col',  'ProbabilityNormalizationColumn','ProbabilityNormalizationCol'     } )
% %         error(['NormalizationType is wrong: ',char(string(NormalizationType)) ]);
% %     end   
   if ~exist('isdebug','var') || isempty (isdebug)
        isdebug = false;  
    end   
    %  
    [N_gene, N_disease] = size( AdjGfD );
    if isempty( AdjDfD )
       AdjDfD = speye(N_disease);  
       warning('AdjDfD is empty.');
    end 
    %
    if ~exist('NormalizationType','var') || isempty(NormalizationType)
        NormalizationType = 'None'; 
    end
    %  
    IsNormalized = true; 
    switch lower( NormalizationType )
        case lower( {'None'} )
            combMatrix = [ AdjGfG, AdjGfD; AdjGfD', AdjDfD    ] ;
            IsNormalized = false;
            
        case lower( {'Weight'} )  
            combMatrix = [ (1-pro_jump).*AdjGfG, pro_jump.*AdjGfD; pro_jump.*AdjGfD', (1-pro_jump).*AdjDfD    ] ;
            IsNormalized = false;
            
        case lower( {'col','ProbabilityNormalizationColumn','NormalizationColumn', 'Column'} )   %概率解释  ，确保列和为1 
            idxDis_WithDiseaseGene =  sum( AdjGfD, 1)~=0;   % mark diseases with disease-genes
            idxGene_WithDisease    = (sum( AdjGfD, 2)~=0)';   % mark genes that are associated with diseases
            % WAdj = getNormalizedMatrix(Adj, NormalizationType, SetIsolatedNodeSelfLoop )
            M_GfG = getNormalizedMatrix(AdjGfG   , NormalizationType, true  ); 
            M_DfD = getNormalizedMatrix(AdjDfD   , NormalizationType, true  ); 
            M_GfD = getNormalizedMatrix(AdjGfD   , NormalizationType, false );  % probabilities from disease space to gene space 
            M_DfG = getNormalizedMatrix(AdjGfD'  , NormalizationType, false );  % probabilities from gene space to disease space
            %
            M_GfG(:,idxGene_WithDisease)       = (1-pro_jump).*M_GfG(:,idxGene_WithDisease); 
            M_DfD(:,idxDis_WithDiseaseGene )   = (1-pro_jump).*M_DfD(:,idxDis_WithDiseaseGene ) ; 
            M_GfD                           = pro_jump.*M_GfD; % Disease-columns without disease-genes is all zeros. So no use idxDis_WithDiseaseGene
            M_DfG                           = pro_jump.*M_DfG; % Gene-columns without diseases is all zeros. So no use idxGene_WithDisease
            %
            combMatrix = [ M_GfG, M_GfD; M_DfG, M_DfD    ] ; 
            
        case lower( {'row','ProbabilityNormalizationRow','NormalizationRow'} )
            idxDis_WithDiseaseGene = (sum( AdjGfD, 1)~=0);   % mark diseases with disease-genes
            idxGene_WithDisease    = (sum( AdjGfD, 2)~=0);   % mark genes that are associated with diseases
            % WAdj = getNormalizedMatrix(Adj, NormalizationType, SetIsolatedNodeSelfLoop )
            M_GfG = getNormalizedMatrix(AdjGfG   , NormalizationType, true ); 
            M_DfD = getNormalizedMatrix(AdjDfD   , NormalizationType, true ); 
            M_GfD = getNormalizedMatrix(AdjGfD   , NormalizationType, false );  % probabilities from disease space to gene space 
            M_DfG = getNormalizedMatrix(AdjGfD'  , NormalizationType, false );  % probabilities from gene space to disease space
            %
            M_GfG(idxGene_WithDisease,:)       = (1-pro_jump).*M_GfG(idxGene_WithDisease,:) ; 
            M_DfD(idxDis_WithDiseaseGene,: )   = (1-pro_jump).*M_DfD(idxDis_WithDiseaseGene,: ) ; 
            M_GfD                           = pro_jump.*M_GfD; % Disease-columns without disease-genes is all zeros. So no use idxDis_WithDiseaseGene
            M_DfG                           = pro_jump.*M_DfG; % Gene-columns without diseases is all zeros. So no use idxGene_WithDisease
            %
            combMatrix = [ M_GfG, M_GfD; M_DfG, M_DfD    ] ;            
            
        case lower( {'LaplacianNormalization'} )
            % 对称化laplacian正则化，对网络节点smooth�?
            % 在异构网络中相当于两个原（对称相似�?）矩阵内部的smooth约束�?(1-pro_jump)�?bipartite矩阵对层间关联的约束�?pro_jump�?
            % 因此 �?��的矩�?等价�?两个normalized矩阵的加权组�?
            % WAdj = getNormalizedMatrix(Adj, NormalizationType, SetIsolatedNodeSelfLoop )
            M_GfG = getNormalizedMatrix(AdjGfG   , NormalizationType, false );   
            M_DfD = getNormalizedMatrix(AdjDfD   , NormalizationType, false ); 
            M_GfD = getNormalizedMatrix(AdjGfD   , NormalizationType, false );  % probabilities from disease space to gene space 
            M_DfG = getNormalizedMatrix(AdjGfD'  , NormalizationType, false );  % probabilities from gene space to disease space
            %
            combMatrix = [ (1-pro_jump).*M_GfG, pro_jump.*M_GfD; pro_jump.*M_DfG, (1-pro_jump).*M_DfD    ] ;            
            
        otherwise
            error('No definition.');
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [Pt, WAdj ]= A_RWRplus(Adj, r_restart, P0, N_max_iter, Eps_min_change, IsNormalized,  NormalizationType)
% A_RWRplus is a generalization of RWR algorithm.
% Including various propagation algorihtms with initial regularization: classical RWR, Label propagation,and so on
% Including a Solver_IterationPropagation, which can be used directly when IsNormalized is TRUE.
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% need getNormalizedMatrix  
% Input 
% Adj
% r_restart
% P0   It should be normalized, though it is neccesary. 
% N_max_iter
% Eps_min_change
% IsNormalized   Whether Adj has been normalized: True or False  
% NormalizationType: including two types of methods by different Normalization Types
% (1) random walk with restart {'ProbabilityNormalizationColumn','ProbabilityNormalizationCol','col','column'}
% (2) 'LaplacianNormalization'   similar to PRINCE(PLOS Computational Biology, 2010, 6: e1000641.)               
% it is equivalent to PRINCE if assigned extended P0 with % disease simialrity and logistic function
% (3)label propagation with memory restart { 'row' ,'ProbabilityNormalizationRow'} 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% Ouput
% Pt  
% WAdj   normalized Adj 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % By Ju Xiang, 
% % % Email: xiang.ju@foxmail.com, xiangju@csu.edu.cn  
% % % 2019-3-11   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % Adj = rand(5); Adj([1 2 ],:)=0.1;   Adj(:, [1 2 ])=0.1;  P0 = [0 0 1 0 0 ]';r_restart=0.7; NormalizationType ='col';
% % Adj = rand(5);   P0 = [0 0 1 0 0 ]';r_restart=0.7; NormalizationType ='col';
% % % Adj = rand(5);   P0 = [0 0 1 0 0; 0 0 0.5 0.5 0 ]';r_restart=0.7; NormalizationType ='col';  
% % % % % % % % % % % % % % % 
    if ~exist('N_max_iter','var') || isempty(N_max_iter) || (isnumeric( N_max_iter) && N_max_iter<=1 ) 
        N_max_iter =100; 
    elseif ~isnumeric( N_max_iter)  
        error('N_max_iter should be isnumeric!!!!!') ;
    end
    %
    if ~exist('Eps_min_change','var') || isempty(Eps_min_change) 
        Eps_min_change =10^-6; 
    elseif isnumeric( Eps_min_change) && Eps_min_change>=1 
        warning('The Eps_min_change is nomenaning. Reset Eps_min_change to be 10^-6.'); 
        Eps_min_change =10^-6;  
    elseif ~isnumeric( Eps_min_change)  
        error('Eps_min_change should be isnumeric!!!!!') ;
    end
    
    if ~exist('IsNormalized','var') || isempty(IsNormalized) 
        IsNormalized = false;  % Adj has been normalized for fast run.   
    end
    
    if ~exist('NormalizationType','var') || isempty(NormalizationType) 
        NormalizationType = 'ProbabilityNormalizationColumn'; %%for 'Random Walk' RWR, RWRH  RWRM  RWRMH and more   
    end        
    % % % 取消�?��矩阵的转换，在调用的外部根据情况指定矩阵形式,除非极端情况 % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %    
    P0  = full(P0); 
    % 
    % AdjIsSparse = isparse(Adj); 
    AdjDensity =nnz(Adj)/numel(Adj); 
    if  (size(P0,2)==1 && AdjDensity<0.3 ) || (size(P0,2)>1 && AdjDensity<0.05 )
        Adj = sparse(Adj);          
    elseif (size(P0,2)==1 && AdjDensity>0.3 ) || (size(P0,2)>1 && AdjDensity>0.05 )
        Adj = full(Adj); 
    else 
        % no operation 
    end
    %
    if IsNormalized 
        WAdj = Adj; 
    else
        % WAdj = getNormalizedMatrix(Adj, 'col', true );
        switch NormalizationType
            case {'ProbabilityNormalizationColumn','ProbabilityNormalizationCol','col','column'}
                % random walk with restart
                WAdj = getNormalizedMatrix(Adj, 'col', true );
                %%%P0 = P0./(sum(P0,1)+eps);    % total probability is 1. 
                
            case 'LaplacianNormalization'  
                % propagation similar to PRINCE(PLOS Computational Biology, 2010, 6: e1000641.)
                % it is equivalent to PRINCE if assigned extended P0 with
                % disease simialrity and logistic function
                % A_PRINCEplus is better. 
                WAdj = getNormalizedMatrix(Adj, 'LaplacianNormalization', true );   
                
            case { 'row' ,'ProbabilityNormalizationRow'} 
                % label propagation with memory restart  
                WAdj = getNormalizedMatrix(Adj, 'row', true );  
                
            otherwise
                error(['NormalizationType is wrong: ',char( string(NormalizationType) )]); 
        end        
    end   
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %     
    % % Solver_IterationPropagation
    % % It can be used directly when IsNormalized is TRUE.  
    Pt = P0;
    for T = 1: N_max_iter
        Pt1 = (1-r_restart)*WAdj*Pt + r_restart*P0;
        if all( sum( abs( Pt1-Pt )) < Eps_min_change )
            break;
        end
        Pt = Pt1;
    end
    Pt = full(Pt); 
end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %      
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %      

