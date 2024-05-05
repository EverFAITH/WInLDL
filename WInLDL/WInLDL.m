function [run_time, results] = WInLDL(seed,data)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code for the ADMM algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close hidden all;
% clear;
rng(seed)
%data = {'Human_Gene', 'Movie', 'Natural_Scene', 'SBU_3DFE', 'SJAFFE', 'Emotion6', 'Fbp5500', 'Flickr_ldl', 'RAF_ML','SCUT_FBP'};
results = zeros(length(data),5);
results_all = zeros(length(data),5);
run_time = zeros(length(data),1);
for i = 1 : length(data)
    %% Load and generate data
    data_name = data{i}
    obr_rate = 0.5
    mu = 2
    load(['data/' data_name '/my_run_obrT=' num2str(obr_rate) '_run=' num2str(seed)  '.mat'])
%   test_rate = 0.2; % 10% data are test data
%   obr_rate = 0.5;  % in training data, 50% labels are missing
    
    %     if exist(['incomp_data\' data_name '_incomplete_' num2str(obr_rate) 'seed=' num2str(seed) '.mat'],'file')
    %         load(['incomp_data\' data_name '_incomplete_' num2str(obr_rate) 'seed=' num2str(seed) '.mat']);
    %         disp('data already exist!')
    %     else
%     load(['data\' data_name '.mat']);
  

    
    
    %% Generated data description
    
    % features: n*d instances' features
    % labels: n*m ground truth labels
    % obrT: n*m 0/1 matrix, 1 means the corresponding position in labels is observed
    
    %train_index = setdiff(1:size(labels,1),test_idx);
 
    D0 = obrT_train .* labels(train_idx,:);
    %D0 = obrT .* labels;
    D0 = D_for_missing(D0);
    %D0 = P_for_missing(D0);
    
%     row_num_obv = sum((D0~=0),2);
%     row_sample = (row_num_obv) ./ (max(row_num_obv));
%     
%     clo_num_obv = sum((D0~=0),1);
%     clo_num_obv(clo_num_obv==0)=1;
    %mean_obv = sum(D0,1) ./ clo_num_obv;
    %mean_obv =  mean_obv ./ sum(mean_obv);
    
    
    mean_obv = mean(D0,1);   
    %mean_obv =  mean_obv ./ sum(mean_obv);
    Du = mean_obv .* (D0==0);
    %[aa,bb] = size(labels(train_idx,:));
    %     Lamda = zeros(size(labels));
    %     [aa,bb] = size(labels);
    %Yt = randerr(aa, bb, 1);
    %Yt = P_for_missing(D);
    %     Yt = D;
    %     cc = sum(Yt,2);
    %     cc(cc==0)=1;
    %Yt = rand(size(labels));
    %Yt = Yt ./ cc;
    %Wt = zeros(size(X_train,2), size(labels,2));
    %Yt = zeros(size(labels(train_idx,:)));
    
    if size(features,2) > 250
        [~,score,~] = pca(features);
        features = score(:,1 : 100);
    end

    if size(features,1) > 500
        features = Normalize_row(features);
    end
    
    X_all = [features,ones(size(features,1),1)];
    X_train = X_all(train_idx,:);
    X_test = X_all(test_idx,:);
    Lamda = zeros(size(labels(train_idx,:)));

    Wt = rand(size(X_train,2), size(labels,2));
    Yt = X_train * Wt;
    Yt = Yt ./ sum(Yt,2);
    
    Pt = (2.^(ones(size(D0)) - D0) .* (D0~=0));
    Pu = ((ones(size(Du)) - Du) .* (D0==0));
    P = (Pt + Pu);
    
    
    eval(['diary', ' record_run', '.txt'])
    diary on
    
    %% Optimization
    iter = 0;
    maxIter = 50;
    tol = 1e-4;
    %mu = 2
    
    
    f_value_Ori = zeros(1,maxIter);
    f_value_ADMM = zeros(1,maxIter);
    
    tic
    w_bar = waitbar(0,'Start Optimization...');
    
    while iter < maxIter
        t1 = toc;
        iter = iter + 1;
        
        W = ((X_train' * X_train + 1e-5 * eye(size(X_train,2)))) \ (X_train' * (Yt - Lamda / mu));   %%% update W
        
        err_W = max(max(abs(W - Wt)));
        Wt = W;
        
        
        D = X_train * W;
        D = projProbSimplex(D);
        D(D0~=0)=D0(D0~=0);
        Dt = D;
        
        Y = (mu * X_train * W + Lamda + P .* P .* Dt) ./ (P .* P  + mu);    %%%  update Y
        Y = projProbSimplex(Y);
        err_Y = max(max(abs(Y - Yt)));
        Yt = Y;
        
        a = 1 + (2 - 1) / 50 * (iter - 1);
        Pu = (a.^(ones(size(Du)) - Du) .* (D0==0));    %%% update Pu
        P = Pt + Pu;
        
        
        err = max(err_W,err_Y);
        
        if err < tol
            break;
        else                                             
            Lamda = Lamda + mu * (X_train * Wt - Yt);   %%% update Lamda
        end
        
        run_time(i,1) = toc;
        
        t2 = toc;
        Str = ['Optimizing, wait please...',num2str(100 * iter / maxIter),'%.',' Time remaining...', num2str((t2 - t1) * (maxIter - iter)),'s'];
        waitbar(iter / maxIter, w_bar, Str)
        
        f_value_Ori(iter) = 1/2 * norm(P.* (X_train * W - Dt),'fro')^2;
        f_value_ADMM(iter) = 1/2 * norm(P.* (Y - Dt),'fro')^2 + trace(Lamda' * (X_train * W - Y)) + mu / 2 * norm((X_train * W - Y),'fro')^2;
    end
    
    
    ErrYD = (X_train * W - Y);

    
    disp(['err_W = ' num2str(err_W)]);
    disp(['err_Y = ' num2str(err_Y)]);
   
    Y = X_all * W;
    Y = projProbSimplex(Y);
    Y(Y==0)=1e-6;

    
    Y_test = Y(test_idx,:);
    
    %save([data_name '_Y' '.mat'],'Y','err_W','err_Y');
    
    figure;
    f_value_Ori(f_value_Ori==0)=[];
    plot(f_value_Ori)
    title(['Original object function loss ', 'mu = ', num2str(mu)])
    
    figure;
    f_value_ADMM(f_value_ADMM==0)=[];
    plot(f_value_ADMM)
    title(['Surrogate object function loss ', 'mu = ', num2str(mu)])
    
    disp('Running MY_IncomLDL-ADMM');
    disp('');

    %% Test Phase
    addpath('measures')

    Cosine_My_admm = cosine(labels(test_idx,:),Y_test);
    Cosine_all_My_admm = cosine(labels,Y);
    disp(['Cosine on test data for My_IncomLDL-admm: ' num2str(Cosine_My_admm)]);
    disp(['Cosine on all data for My_IncomLDL-admm: ' num2str(Cosine_all_My_admm)]);
    

    Intersection_My_admm = intersection(labels(test_idx,:),Y_test);
    Intersection_all_My_admm = intersection(labels,Y);
    disp(['Intersection on test data for My_IncomLDL-admm: ' num2str(Intersection_My_admm)]);
    disp(['Intersection on all data for My_IncomLDL-admm: ' num2str(Intersection_all_My_admm)]);
       

    Chebyshev_My_admm = chebyshev(labels(test_idx,:),Y_test);
    Chebyshev_all_My_admm = chebyshev(labels,Y);
    disp(['Chebyshev on test data for My_IncomLDL-admm: ' num2str(Chebyshev_My_admm)]);
    disp(['Chebyshev on all data for My_IncomLDL-admm: ' num2str(Chebyshev_all_My_admm)]);
    

    Clark_My_admm = clark(labels(test_idx,:),Y_test);
    Clark_all_My_admm = clark(labels,Y);
    disp(['Clark on test data for My_IncomLDL-admm: ' num2str(Clark_My_admm)]);
    disp(['Clark on all data for My_IncomLDL-admm: ' num2str(Clark_all_My_admm)]);
    

    Canberraon_My_admm = canberra(labels(test_idx,:),Y_test);
    Canberraon_all_My_admm = canberra(labels,Y);
    disp(['Canberraon on test data for My_IncomLDL-admm: ' num2str(Canberraon_My_admm)]);
    disp(['Canberraon on all data for My_IncomLDL-admm: ' num2str(Canberraon_all_My_admm)]);
    
    
    results(i,:) = [Cosine_My_admm, Intersection_My_admm, Chebyshev_My_admm, Clark_My_admm, Canberraon_My_admm];
    results_all(i,:) = [Cosine_all_My_admm, Intersection_all_My_admm, Chebyshev_all_My_admm, Clark_all_My_admm, Canberraon_all_My_admm];
    soft_results = results;
    soft_results_all = results_all;
    save(['my_results_save/my_results_10_mu=' num2str(mu) '_obrT=' num2str(obr_rate) '.mat'], 'soft_results');
    save(['my_results_save/my_results_all_10_mu=' num2str(mu) '_obrT=' num2str(obr_rate) '_seed=' num2str(seed) '.mat'], 'soft_results_all');
end

diary off



