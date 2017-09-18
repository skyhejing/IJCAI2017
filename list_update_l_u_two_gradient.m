function g_i = list_update_l_u_two_gradient( l_u,u_l,trainset_three,matrix_feature,lamda,i )
    %first
    u_l_update=u_l(trainset_three(i==trainset_three(:,3),1),:);
    l_u_one=l_u(trainset_three(i==trainset_three(:,3),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,3),6),:);
    l_u_three=l_u(trainset_three(i==trainset_three(:,3),7),:);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    x_three=sum(u_l_update .* l_u_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_one_first=(u_l_update .* exp(x_one_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times))-u_l_update;
    result_one=result_one_first+ lamda .* l_u_one;
    result_one=sum(result_one,1);
    
    %second
    u_l_update=u_l(trainset_three(i==trainset_three(:,6),1),:);
    l_u_one=l_u(trainset_three(i==trainset_three(:,6),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,6),6),:);
    l_u_three=l_u(trainset_three(i==trainset_three(:,6),7),:);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    x_three=sum(u_l_update .* l_u_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_two_first=(u_l_update .* exp(x_two_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times));
    result_two_second=(u_l_update .* exp(x_two_times)) ./ (exp(x_two_times)+exp(x_three_times))-u_l_update;
    result_two=result_two_first+result_two_second+ lamda .* l_u_two;
    result_two=sum(result_two,1);
    
    %third
    u_l_update=u_l(trainset_three(i==trainset_three(:,7),1),:);
    l_u_one=l_u(trainset_three(i==trainset_three(:,7),3),:);
    l_u_two=l_u(trainset_three(i==trainset_three(:,7),6),:);
    l_u_three=l_u(trainset_three(i==trainset_three(:,7),7),:);
    
    x_one=sum(u_l_update .* l_u_one,2);
    x_two=sum(u_l_update .* l_u_two,2);
    x_three=sum(u_l_update .* l_u_three,2);
    
    x_one_times=repmat(x_one,[1,matrix_feature]);
    x_two_times=repmat(x_two,[1,matrix_feature]);
    x_three_times=repmat(x_three,[1,matrix_feature]);
    
    result_three_first=(u_l_update .* exp(x_three_times)) ./ (exp(x_one_times)+exp(x_two_times)+exp(x_three_times));
    result_three_second=(u_l_update .* exp(x_three_times)) ./ (exp(x_two_times)+exp(x_three_times));
    result_three=result_three_first+result_three_second+ lamda .* l_u_three;
    result_three=sum(result_three,1);
    
    %all
    result=result_one+result_two+result_three;
    g_i=sum(result,1);
end