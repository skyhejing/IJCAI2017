function g_i = list_update_p_l_two_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i )
    l_p_one=l_p(trainset_three(i==trainset_three(:,2),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,2),6),:);
    l_p_three=l_p(trainset_three(i==trainset_three(:,2),7),:);
    
    p_l_update=repmat(p_l(i,:),[size(l_p_one,1),1]);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    x_three=sum(p_l_update .* l_p_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_first=(l_p_one .* exp(x_one_times) + l_p_two .* exp(x_two_times)+ l_p_three .* exp(x_three_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times))-l_p_one;
    result_second=(l_p_two .* exp(x_two_times)+ l_p_three .* exp(x_three_times)) ./ (exp(x_two_times)+exp(x_three_times))-l_p_two;
    
    result=result_first+result_second+ lamda .* p_l_update;
    
    g_i=sum(result,1);
end