function [f,g] = list_update_l_u_three(l_u,u_l, trainset_three,lamda,h,matrix_feature)
%deal f

l_u=reshape(l_u, h, matrix_feature);

u_l_trainset=u_l(trainset_three(:,1),:);
l_u_trainset_one=l_u(trainset_three(:,3),:);
l_u_trainset_two=l_u(trainset_three(:,6),:);
l_u_trainset_three=l_u(trainset_three(:,9),:);
l_u_trainset_four=l_u(trainset_three(:,10),:);

u_l_sum_one=sum(u_l_trainset .* l_u_trainset_one,2);
u_l_sum_two=sum(u_l_trainset .* l_u_trainset_two,2);
u_l_sum_three=sum(u_l_trainset .* l_u_trainset_three,2);
u_l_sum_four=sum(u_l_trainset .* l_u_trainset_four,2);

result_first=log(exp(u_l_sum_one) + exp(u_l_sum_two) +exp(u_l_sum_three)+exp(u_l_sum_four))-u_l_sum_one;
result_second=log(exp(u_l_sum_two) +exp(u_l_sum_three)+exp(u_l_sum_four))-u_l_sum_two;
result_third=log(exp(u_l_sum_three)+exp(u_l_sum_four))-u_l_sum_three;



f=sum(result_first+result_second+result_third);


%deal g
g = zeros(size(l_u));
%comment at 20151027. Replaced by arrayfun.

l_u_new_cell=arrayfun(@(i) list_update_l_u_three_gradient( l_u,u_l,trainset_three,matrix_feature,lamda,i ),1:size(l_u,1),'UniformOutput', false);
g=reshape(cell2mat(l_u_new_cell),size(l_u'))';
g = g(:);
% g(g<0.00001)=0.00001;
% g(isnan(g))=100;
legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'list_update_l_u_three: g is not legal!'
	end

end %endfunction