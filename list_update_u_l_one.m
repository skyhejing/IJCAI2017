function [f,g] = list_update_u_l_one(u_l_sample, l_u,trainset_three,lamda,h,matrix_feature)
%deal f

u_l_sample=reshape(u_l_sample, h, matrix_feature);

u_l_trainset=u_l_sample(trainset_three(:,1),:);
l_u_trainset_one=l_u(trainset_three(:,3),:);
l_u_trainset_two=l_u(trainset_three(:,4),:);

u_l_sum_one=sum(u_l_trainset .* l_u_trainset_one,2);
u_l_sum_two=sum(u_l_trainset .* l_u_trainset_two,2);

result_first=log(exp(u_l_sum_one) + exp(u_l_sum_two))-u_l_sum_one;

f=sum(result_first);


%deal g
g = zeros(size(u_l_sample));

u_l_new_cell=arrayfun(@(i) list_update_u_l_one_gradient( l_u,u_l_sample,trainset_three,matrix_feature,lamda,i ),1:size(u_l_sample,1),'UniformOutput', false);
g=reshape(cell2mat(u_l_new_cell),size(u_l_sample'))';
g = g(:);
% g(g<0.00001)=0.00001;
% g(isnan(g))=100;
legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'u_l_update: g is not legal!'
	end

end %endfunction