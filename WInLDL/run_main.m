clear;clc;
close hidden all

run = 5;
mu_ = 2
data = {'SBU_3DFE'}
%data = {'Human_Gene', 'Movie', 'Natural_Scene', 'SBU_3DFE', 'SJAFFE', 'Emotion6', 'Fbp5500', 'Flickr_ldl', 'RAF_ML','SCUT_FBP'};
main_results = zeros(length(data),5,5);
run_time = zeros(length(data),5);


for i =  1 : run
    rng(i)
    [run_time(:,i), main_results(:,:,i)] = WInLDL(i,data);
end


main_mean_results = mean(main_results,3);
main_std_results = std(main_results,0,3);

save(['obrT_0.5_main_results_mu=' num2str(mu_) '.mat'], 'main_results')
save(['obrT_0.5_main_mean_results_mu=' num2str(mu_) '.mat'], 'main_mean_results')
save(['obrT_0.5_main_std_results_mu=' num2str(mu_) '.mat'], 'main_std_results')
save(['obrT_0.5_runtime_mu=' num2str(mu_) '.mat'],'run_time')