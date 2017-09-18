function g_i = list_update_l_p_one_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i )
    %first
    p_l_update=p_l(trainset_three(i==trainset_three(:,3),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,3),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,3),4),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    
    result_one_first=(p_l_update .* exp(x_one_times)) ./ (exp(x_one_times)+exp(x_two_times))-p_l_update;
    result_one=result_one_first+ lamda .* l_p_one;
    result_one=sum(result_one,1);
    
    %second
    p_l_update=p_l(trainset_three(i==trainset_three(:,4),2),:);
    l_p_one=l_p(trainset_three(i==trainset_three(:,4),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,4),4),:);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    
    result_two_first=(p_l_update .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times));
    result_two=result_two_first+ lamda .* l_p_two;
    result_two=sum(result_two,1);
    
    %all
    result=result_one+result_two;
    g_i=sum(result,1);
end