function g_i = list_update_l_p_three_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i )
    %first
    p_l_update=p_l(trainset_three(i==trainset_three(:,3),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,3),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,3),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,3),9),:);
    l_p_four=l_p(trainset_three(i==trainset_three(:,3),10),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    x_four=sum(p_l_update .* l_p_four,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    x_four_times=repmat(x_four,[1,matrix_feature]);
    
    
    result_one_first=(p_l_update .* exp(x_one_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times)+exp(x_four_times))-p_l_update;
    result_one=result_one_first+ lamda .* l_p_one;
    result_one=sum(result_one,1);
    
    %second
    p_l_update=p_l(trainset_three(i==trainset_three(:,6),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,6),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,6),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,6),9),:);
    l_p_four=l_p(trainset_three(i==trainset_three(:,6),10),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    x_four=sum(p_l_update .* l_p_four,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    x_four_times=repmat(x_four,[1,matrix_feature]);
    
    result_two_first=(p_l_update .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times)+exp(x_four_times));
    result_two_second=(p_l_update .* exp(x_two_times)) ./ (exp(x_two_times)+exp(x_three_times)+exp(x_four_times))-p_l_update;
    result_two=result_two_first+result_two_second+ lamda .* l_p_two;
    result_two=sum(result_two,1);
    
    %third
    p_l_update=p_l(trainset_three(i==trainset_three(:,9),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,9),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,9),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,9),9),:);
    l_p_four=l_p(trainset_three(i==trainset_three(:,9),10),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    x_four=sum(p_l_update .* l_p_four,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    x_four_times=repmat(x_four,[1,matrix_feature]);
    
    result_three_first=(p_l_update .* exp(x_three_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times)+exp(x_four_times));
    result_three_second=(p_l_update .* exp(x_three_times)) ./ (exp(x_two_times)+exp(x_three_times)+exp(x_four_times));
    result_three_third=(p_l_update .* exp(x_three_times)) ./ (exp(x_three_times)+exp(x_four_times))-p_l_update;
    result_three=result_three_first+result_three_second+result_three_third+ lamda .* l_p_three;
    result_three=sum(result_three,1);
    
    %fourth
    p_l_update=p_l(trainset_three(i==trainset_three(:,10),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,10),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,10),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,10),9),:);
    l_p_four=l_p(trainset_three(i==trainset_three(:,10),10),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    x_four=sum(p_l_update .* l_p_four,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    x_four_times=repmat(x_four,[1,matrix_feature]);
    
    result_four_first=(p_l_update .* exp(x_four_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times)+exp(x_four_times));
    result_four_second=(p_l_update .* exp(x_four_times)) ./ (exp(x_two_times)+exp(x_three_times)+exp(x_four_times));
    result_four_third=(p_l_update .* exp(x_four_times)) ./ (exp(x_three_times)+exp(x_four_times));
    result_four=result_four_first+result_four_second+result_four_third+ lamda .* l_p_four;
    result_four=sum(result_four,1);
    
    %all
    result=result_one+result_two+result_three+result_four;
    g_i=sum(result,1);
end