function [Myresult_1,Myresult_5,Myresult_location_1,Myresult_location_5,Myresult_location_10,Myresult_location_20,Myresult_location_30,Myresult_location_40,Myresult_location_50] = list_prediction_MLE( testset,l_u,u_l,l_p,p_l,i,distance_matrix_frac,firstCategoryLocation,location_unique_test,testset_location_category)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    u_l_predict=u_l(testset(i,1),:);
    
    pre_category=testset(i,11:25);
    pre_category=pre_category';
    pre_category_no_zero=pre_category(pre_category>0);
    p_l_pre_category=p_l(pre_category_no_zero,:);
    p_l_predict=sum(p_l_pre_category,1);
    
    l_u_test=l_u;
    l_p_test=l_p;
    
    u_l_tensor=repmat(u_l_predict,[size(l_u_test,1) 1]);
    p_l_tensor=repmat(p_l_predict,[size(l_p_test,1) 1]);
    
    tensor_result=u_l_tensor .* l_u_test + p_l_tensor .* l_p_test;
    tensor_result_sum=sum(tensor_result,2);
   
    
%     [~, index_sum]=sort(alpha_result_sum,'descend');
    [~, index_sum]=sort(tensor_result_sum,'descend');
    index_sum_top1=index_sum(1);
    index_sum_top5=index_sum(1:5);
%     index_sum_top10=index_sum(1:10);
%     index_sum_top20=index_sum(1:20);
%     index_sum_top30=index_sum(1:30);
%     index_sum_top40=index_sum(1:40);
%     index_sum_top50=index_sum(1:50);
%     index_sum_top60=index_sum(1:60);
%     index_sum_top70=index_sum(1:70);
%     index_sum_top80=index_sum(1:80);
%     index_sum_top90=index_sum(1:90);
%     index_sum_top100=index_sum(1:100);
    first_category_test=1:8;
    first_category_test=first_category_test';
    if intersect(testset(i,27:41),first_category_test(index_sum_top1,1))
        Myresult_1=1;
    else
        Myresult_1=0;
    end
    
    if intersect(testset(i,27:41),first_category_test(index_sum_top5,1))
        Myresult_5=1;
    else
        Myresult_5=0;
    end
    


    %20170214. Xin Li's idea
    %tensor_result_sum(tensor_result_sum<tensor_result_sum(index_sum(200)))=-50;
    tensor_result_sum(tensor_result_sum<tensor_result_sum(index_sum(8)))=-0.000001;
    tensor_result_sum_exp=exp(tensor_result_sum);
    tensor_result_sum_exp_p=bsxfun(@rdivide,tensor_result_sum_exp,sum(tensor_result_sum_exp,1));
    tensor_result_sum_exp_p(isnan(tensor_result_sum_exp_p))=1;
    tensor_result_sum_exp_p(tensor_result_sum_exp_p>1)=1;
    tensor_result_sum_exp_p(tensor_result_sum_exp_p<exp(-50))=exp(-50);
    
    %tensor_result_sum_exp_p(242)=exp(-50);
    tensor_result_sum_exp_p(9)=0.01;
    
    
    %test_testset(:,1)=tensor_result_sum_exp_p(test_testset(:,1));
    testset_location_category_p(:,:)=tensor_result_sum_exp_p(testset_location_category(:,:));
    testset_location_category_p_sum=sum(testset_location_category_p,2);

    %predict the location
    prelocation_line=location_unique_test(location_unique_test(:,1)==testset(i,2),4);
    distance_test_frac=distance_matrix_frac(prelocation_line,:);
    distance_test_frac=distance_test_frac';
    line_numner=1:size(distance_test_frac,1);
    distance_test_frac(:,2)=line_numner';
    
    top_category=first_category_test(index_sum(1:8));
    filter_locations=firstCategoryLocation(top_category,:);
    filter_locations_unique=unique(filter_locations);
    filter_locations_unique(1,:)=[];
    
    candidate_list=distance_test_frac(filter_locations_unique,:);
    
    location_score=10.5 .* candidate_list(:,1);
    
    %20170214 Inspired by Xin Li. Deal with the category ranking
    %influence by a softmax function.
    testset_location_category_p_sum_candidate=testset_location_category_p_sum(filter_locations_unique);
    location_score_softmax=bsxfun(@power,location_score,testset_location_category_p_sum_candidate);
    
    
    %[~, index_location]=sort(location_score,'descend');
    [~, index_location]=sort(location_score_softmax,'descend');
    
    index_location_top1=index_location(1);
    index_location_top5=index_location(1:5);
    index_location_top10=index_location(1:10);
    index_location_top20=index_location(1:20);
    index_location_top30=index_location(1:30);
    index_location_top40=index_location(1:40);
    index_location_top50=index_location(1:50);
    
    if any(testset(i,26)==candidate_list(index_location_top1,2))
        Myresult_location_1=1;
    else
        Myresult_location_1=0;
    end
    
    if any(testset(i,26)==candidate_list(index_location_top5,2))
        Myresult_location_5=1;
    else
        Myresult_location_5=0;
    end
    
    if any(testset(i,26)==candidate_list(index_location_top10,2))
        Myresult_location_10=1;
    else
        Myresult_location_10=0;
    end
    
    if any(testset(i,26)==candidate_list(index_location_top20,2))
        Myresult_location_20=1;
    else
        Myresult_location_20=0;
    end
    
    if any(testset(i,26)==candidate_list(index_location_top30,2))
        Myresult_location_30=1;
    else
        Myresult_location_30=0;
    end
    
    if any(testset(i,26)==candidate_list(index_location_top40,2))
        Myresult_location_40=1;
    else
        Myresult_location_40=0;
    end
    
    if any(testset(i,26)==candidate_list(index_location_top50,2))
        Myresult_location_50=1;
    else
        Myresult_location_50=0;
    end
    
    
    
end