function g_i = list_update_u_l_one_gradient( l_u,u_l,trainset_three,matrix_feature,lamda,i )
    l_u_one=l_u(trainset_three(i==trainset_three(:,1),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,1),4),:);
    
    u_l_update=repmat(u_l(i,:),[size(l_u_one,1),1]);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    
    result_first=(l_u_one .* exp(x_one_times) + l_u_two .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times))-l_u_one;
    
    result=result_first+ lamda .* u_l_update;
    
    g_i=sum(result,1);
end