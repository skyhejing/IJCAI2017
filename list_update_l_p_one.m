function [f,g] = list_update_l_p_one(l_p,p_l, trainset_three,lamda,h,matrix_feature)
%deal f

l_p=reshape(l_p, h, matrix_feature);

p_l_trainset=p_l(trainset_three(:,2),:);
l_p_trainset_one=l_p(trainset_three(:,3),:);
l_p_trainset_two=l_p(trainset_three(:,4),:);

p_l_sum_one=sum(p_l_trainset .* l_p_trainset_one,2);
p_l_sum_two=sum(p_l_trainset .* l_p_trainset_two,2);

result_first=log(exp(p_l_sum_one) + exp(p_l_sum_two))-p_l_sum_one;

f=sum(result_first);


%deal g
g = zeros(size(l_p));
l_p_new_cell=arrayfun(@(i) list_update_l_p_one_gradient( l_p,p_l,trainset_three,matrix_feature,lamda,i ),1:size(l_p,1),'UniformOutput', false);
g=reshape(cell2mat(l_p_new_cell),size(l_p'))';
g = g(:);
% g(g<0.00001)=0.00001;
% g(isnan(g))=100;
legal = sum(any(imag(g(:))))==0 & sum(isnan(g(:)))==0 & sum(isinf(g(:)))==0;
	if ~legal
	    disp 'l_p_update: g is not legal!'
	end

end %endfunction