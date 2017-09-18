function g_i = list_update_l_u_one_gradient( l_u,u_l,trainset_three,matrix_feature,lamda,i )
    %first
    u_l_update=u_l(trainset_three(i==trainset_three(:,3),1),:);
    l_u_one=l_u(trainset_three(i==trainset_three(:,3),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,3),4),:);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    
    result_one_first=(u_l_update .* exp(x_one_times)) ./ (exp(x_one_times)+exp(x_two_times))-u_l_update;
    result_one=result_one_first+ lamda .* l_u_one;
    result_one=sum(result_one,1);
    
    %second
    u_l_update=u_l(trainset_three(i==trainset_three(:,4),1),:);
    l_u_one=l_u(trainset_three(i==trainset_three(:,4),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,4),4),:);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    
    result_two_first=(u_l_update .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times));
    result_two=result_two_first+ lamda .* l_u_two;
    result_two=sum(result_two,1);
    
    %all
    result=result_one+result_two;
    g_i=sum(result,1);
end