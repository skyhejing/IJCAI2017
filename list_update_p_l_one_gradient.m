function g_i = list_update_p_l_one_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i )
    l_p_one=l_p(trainset_three(i==trainset_three(:,2),3),:);
    l_p_two=l_p(trainset_three(i==trainset_three(:,2),4),:);
    
    p_l_update=repmat(p_l(i,:),[size(l_p_one,1),1]);
    
    x_one=sum(p_l_update .* l_p_one,2);
    x_two=sum(p_l_update .* l_p_two,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    
    result_first=(l_p_one .* exp(x_one_times) + l_p_two .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times))-l_p_one;
    
    result=result_first+ lamda .* p_l_update;
    
    g_i=sum(result,1);
end