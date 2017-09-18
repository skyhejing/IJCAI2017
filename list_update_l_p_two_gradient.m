function g_i = list_update_l_p_two_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i )
    %first
    p_l_update=p_l(trainset_three(i==trainset_three(:,3),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,3),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,3),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,3),7),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_one_first=(p_l_update .* exp(x_one_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times))-p_l_update;
    result_one=result_one_first+ lamda .* l_p_one;
    result_one=sum(result_one,1);
    
    %second
    p_l_update=p_l(trainset_three(i==trainset_three(:,6),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,6),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,6),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,6),7),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_two_first=(p_l_update .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times));
    result_two_second=(p_l_update .* exp(x_two_times)) ./ (exp(x_two_times)+exp(x_three_times))-p_l_update;
    result_two=result_two_first+result_two_second+ lamda .* l_p_two;
    result_two=sum(result_two,1);
    
    %third
    p_l_update=p_l(trainset_three(i==trainset_three(:,7),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,7),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,7),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,7),7),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_three_first=(p_l_update .* exp(x_three_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times));
    result_three_second=(p_l_update .* exp(x_three_times)) ./ (exp(x_two_times)+exp(x_three_times));
    result_three=result_three_first+result_three_second+ lamda .* l_p_three;
    result_three=sum(result_three,1);

    %all
    result=result_one+result_two+result_three;
    g_i=sum(result,1);
end