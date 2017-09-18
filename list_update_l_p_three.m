function [f,g] = list_update_l_p_three(l_p,p_l, trainset_three,lamda,h,matrix_feature)
%deal f

l_p=reshape(l_p, h, matrix_feature);

p_l_trainset=p_l(trainset_three(:,2),:);
l_p_trainset_one=l_p(trainset_three(:,3),:);
l_p_trainset_two=l_p(trainset_three(:,6),:);
l_p_trainset_three=l_p(trainset_three(:,9),:);
l_p_trainset_four=l_p(trainset_three(:,10),:);

p_l_sum_one=sum(p_l_trainset .* l_p_trainset_one,2);
p_l_sum_two=sum(p_l_trainset .* l_p_trainset_two,2);
p_l_sum_three=sum(p_l_trainset .* l_p_trainset_three,2);
p_l_sum_four=sum(p_l_trainset .* l_p_trainset_four,2);

result_first=log(exp(p_l_sum_one) + exp(p_l_sum_two) +exp(p_l_sum_three)+exp(p_l_sum_four))-p_l_sum_one;
result_second=log(exp(p_l_sum_two) +exp(p_l_sum_three)+exp(p_l_sum_four))-p_l_sum_two;
result_third=log(exp(p_l_sum_three)+exp(p_l_sum_four))-p_l_sum_three;

f=sum(result_first+result_second+result_third);


%deal g
g = zeros(size(l_p));
l_p_new_cell=arrayfun(@(i) list_update_l_p_three_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i ),1:size(l_p,1),'UniformOutput', false);
g=reshape(cell2mat(l_p_new_cell),size(l_p'))';
g = g(:);
% g(g<0.00001)=0.00001;
% g(isnan(g))=100;
legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'l_p_update: g is not legal!'
	end

end %endfunction