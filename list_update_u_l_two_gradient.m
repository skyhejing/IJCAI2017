function g_i = list_update_u_l_two_gradient( l_u,u_l,trainset_three,matrix_feature,lamda,i )
    l_u_one=l_u(trainset_three(i==trainset_three(:,1),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,1),6),:);
    l_u_three=l_u(trainset_three(i==trainset_three(:,1),7),:);
    
    u_l_update=repmat(u_l(i,:),[size(l_u_one,1),1]);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    x_three=sum(u_l_update .* l_u_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_first=(l_u_one .* exp(x_one_times) + l_u_two .* exp(x_two_times)+ l_u_three .* exp(x_three_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times))-l_u_one;
    result_second=(l_u_two .* exp(x_two_times)+ l_u_three .* exp(x_three_times)) ./ (exp(x_two_times)+exp(x_three_times))-l_u_two;

    result=result_first+result_second+ lamda .* u_l_update;
    
    g_i=sum(result,1);
end