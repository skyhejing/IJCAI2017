addpath(genpath(pwd));
lamda=0.01;    %init the lamda
matrix_feature=10;    %the number of matrix feature
maxFunEvals = 2; %Result after maxFunEvals evaluations of limited-memory solvers


num_sample_one=1; % the amount of trainset_one
num_sample_two=1;  %the amount of trainset_two
num_sample_three=1;  %the amount of trainset_three

%The number of iteration.
iteration_i=1;
iteration_one=100;
iteration_two=50;
iteration_three=10;

%init tensor parameter
u_l=normrnd(0,0.0000025,[size(user_unique,1),matrix_feature]);
l_u=normrnd(0,0.0000025,[8,matrix_feature]);

p_l=normrnd(0,0.0000025,[8,matrix_feature]);
l_p=normrnd(0,0.0000025,[8,matrix_feature]);

u_l_new=normrnd(0,0.0000025,[size(user_unique,1),matrix_feature]);
l_u_new=normrnd(0,0.0000025,[8,matrix_feature]);

p_l_new=normrnd(0,0.0000025,[8,matrix_feature]);
l_p_new=normrnd(0,0.0000025,[8,matrix_feature]);



[row_testset, col_testset]=size(testset);
row_trainset_one=size(trainset_one,1);
row_trainset_two=size(trainset_two,1);
row_trainset_three=size(trainset_three,1);

        %categories id
        testset_location_category_orginal(:,2:16)=testset_orginal(:,11:25);
        %location id
        testset_location_category_orginal(:,1)=testset_orginal(:,2);
        testset_location_category_unique=unique(testset_location_category_orginal,'rows');
        testset_location_category_nounique=testset_location_category_unique(:,2:16);
        
        [row_testset_location_category, col_testset_location_category]=size(testset_location_category_nounique);
        testset_location_category=zeros(row_testset_location_category, col_testset_location_category);
        for i_testset_location_category=1:row_testset_location_category
            unique_yuansu=unique(testset_location_category_nounique(i_testset_location_category,:));
            row_unique_yuansu=size(unique_yuansu,2);
            testset_location_category(i_testset_location_category,1:row_unique_yuansu)=unique_yuansu;
        end
        testset_location_category(testset_location_category==0)=9;


fid=fopen('result.txt','a+');
while 1
    iteration_i=iteration_i+1;
    
    %setting for minFunc20160113
    options = [];
    options.display = 'none';
    options.Method = 'lbfgs';
    options.maxFunEvals = maxFunEvals;
    
    
    for i_three=1:iteration_three
    %trainset_three
    %give every train sample a negative sample. This negative sample must be a
    %location line number.
    rand_trainset_three_number=randsample(size(trainset_three,1),num_sample_three);
    rand_trainset_three=trainset_three(rand_trainset_three_number,:);
    [row_rand_trainset_three,col_rand_trainset_three]=size(rand_trainset_three);
    rand_negative_three=randsample(8,row_rand_trainset_three,'true');
    rand_trainset_three(:,10)=rand_negative_three;
    
    %test the result value
    f_u_l_three =result_value_u_l_three( u_l,l_u,rand_trainset_three);
    f_p_l_three =result_value_p_l_three( p_l,l_p,rand_trainset_three);
   
    
    %update l_u of trainset_three with first order
    l_u_unique_three_first=unique(rand_trainset_three(:,3),'stable');
    l_u_unique_three_second=unique(rand_trainset_three(:,6),'stable');
    l_u_unique_three_third=unique(rand_trainset_three(:,9),'stable');
    l_u_unique_three_fourth=unique(rand_trainset_three(:,10),'stable');
    l_u_unique_three=unique([l_u_unique_three_first;l_u_unique_three_second;l_u_unique_three_third;l_u_unique_three_fourth],'stable');
    
    l_u_three_sample=l_u(l_u_unique_three,:);
    trainset_three_sample=rand_trainset_three;
    l_u_three = containers.Map(l_u_unique_three(:,1),1:size(l_u_unique_three,1));
    
    locationline_new=arrayfun(@(i) l_u_three(trainset_three_sample(i,3)),1:size(trainset_three_sample,1));
    trainset_three_sample(:,3)=locationline_new';
    locationline_new=arrayfun(@(i) l_u_three(trainset_three_sample(i,6)),1:size(trainset_three_sample,1));
    trainset_three_sample(:,6)=locationline_new';
    locationline_new=arrayfun(@(i) l_u_three(trainset_three_sample(i,9)),1:size(trainset_three_sample,1));
    trainset_three_sample(:,9)=locationline_new';
    locationline_new=arrayfun(@(i) l_u_three(trainset_three_sample(i,10)),1:size(trainset_three_sample,1));
    trainset_three_sample(:,10)=locationline_new';
%     opt_l_u_first_three = minFunc(@list_update_l_u_three,l_u(:),options,u_l,trainset_three,lamda,size(location_unique,1),matrix_feature);
    opt_l_u_three = minFunc(@list_update_l_u_three,l_u_three_sample(:),options,u_l,trainset_three_sample,lamda,size(l_u_unique_three,1),matrix_feature);
    l_u_sample_new=reshape(opt_l_u_three, size(l_u_unique_three,1), matrix_feature);
    l_u(l_u_unique_three,:)=l_u_sample_new;
    %test the result value
    f_u_l_three =result_value_u_l_three( u_l,l_u,rand_trainset_three);
    
    
        %update l_p of trainset_three with first order
    l_p_three_sample=l_p(l_u_unique_three,:);
    opt_l_p_three = minFunc(@list_update_l_p_three,l_p_three_sample(:),options,p_l,trainset_three_sample,lamda,size(l_u_unique_three,1),matrix_feature);
    l_p_sample_new=reshape(opt_l_p_three, size(l_u_unique_three,1), matrix_feature);
    l_p(l_u_unique_three,:)=l_p_sample_new;
    %test the result value
    f_p_l_three =result_value_p_l_three( p_l,l_p,rand_trainset_three);



    
     %update u_l of trainset_three
    u_l_unique_three=unique(rand_trainset_three(:,1),'stable');
    u_l_three_sample=u_l(u_l_unique_three,:);
    trainset_three_sample=rand_trainset_three;
    u_l_three = containers.Map(u_l_unique_three(:,1),1:size(u_l_unique_three,1));
    userline_new=arrayfun(@(i) u_l_three(trainset_three_sample(i,1)),1:size(trainset_three_sample,1));
    trainset_three_sample(:,1)=userline_new';
%     opt_u_l_three = minFunc(@list_update_u_l_three,u_l_sample(:),options,user_unique_three,l_u,trainset_three,lamda,size(user_unique,1),matrix_feature);
    opt_u_l_three = minFunc(@list_update_u_l_three,u_l_three_sample(:),options,l_u,trainset_three_sample,lamda,size(u_l_unique_three,1),matrix_feature);
    u_l_sample_new=reshape(opt_u_l_three, size(u_l_unique_three,1), matrix_feature);
    u_l(u_l_unique_three,:)=u_l_sample_new;
    %test the result value
    f_u_l_three =result_value_u_l_three( u_l,l_u,rand_trainset_three);
    
         %update p_l of trainset_three
    p_l_unique_three=unique(rand_trainset_three(:,2),'stable');
    p_l_three_sample=p_l(p_l_unique_three,:);
    trainset_three_sample=rand_trainset_three;
    p_l_three = containers.Map(p_l_unique_three(:,1),1:size(p_l_unique_three,1));
    locationline_new=arrayfun(@(i) p_l_three(trainset_three_sample(i,2)),1:size(trainset_three_sample,1));
    trainset_three_sample(:,2)=locationline_new';
    opt_p_l_three = minFunc(@list_update_p_l_three,p_l_three_sample(:),options,l_p,trainset_three_sample,lamda,size(p_l_unique_three,1),matrix_feature);
    p_l_sample_new=reshape(opt_p_l_three, size(p_l_unique_three,1), matrix_feature);
    p_l(p_l_unique_three,:)=p_l_sample_new;
    %test the result value
    f_p_l_three =result_value_p_l_three( p_l,l_p,rand_trainset_three);
    
    end

    
    
    for i_two=1:iteration_two
        %trainset_two
    %give every train sample a negative sample. This negative sample must be a
    %location line number.
    rand_trainset_two_number=randsample(size(trainset_two,1),num_sample_two);
    rand_trainset_two=trainset_two(rand_trainset_two_number,:);
    [row_rand_trainset_two,col_rand_trainset_two]=size(rand_trainset_two);
    rand_negative_two=randsample(8,row_rand_trainset_two,'true');
    
    rand_trainset_two(:,7)=rand_negative_two;
    
    %test the result value
    f_u_l_two =result_value_u_l_two( u_l,l_u,rand_trainset_two);
    f_p_l_two =result_value_p_l_two( p_l,l_p,rand_trainset_two);

    %update l_u of trainset_two with first order
    l_u_unique_two_first=unique(rand_trainset_two(:,3),'stable');
    l_u_unique_two_second=unique(rand_trainset_two(:,6),'stable');
    l_u_unique_two_third=unique(rand_trainset_two(:,7),'stable');
    l_u_unique_two=unique([l_u_unique_two_first;l_u_unique_two_second;l_u_unique_two_third],'stable');
    
    l_u_two_sample=l_u(l_u_unique_two,:);
    trainset_two_sample=rand_trainset_two;
    l_u_two = containers.Map(l_u_unique_two(:,1),1:size(l_u_unique_two,1));
    
    locationline_new=arrayfun(@(i) l_u_two(trainset_two_sample(i,3)),1:size(trainset_two_sample,1));
    trainset_two_sample(:,3)=locationline_new';
    locationline_new=arrayfun(@(i) l_u_two(trainset_two_sample(i,6)),1:size(trainset_two_sample,1));
    trainset_two_sample(:,6)=locationline_new';
    locationline_new=arrayfun(@(i) l_u_two(trainset_two_sample(i,7)),1:size(trainset_two_sample,1));
    trainset_two_sample(:,7)=locationline_new';
    opt_l_u_two = minFunc(@list_update_l_u_two,l_u_two_sample(:),options,u_l,trainset_two_sample,lamda,size(l_u_unique_two,1),matrix_feature);
    l_u_sample_new=reshape(opt_l_u_two, size(l_u_unique_two,1), matrix_feature);
    l_u(l_u_unique_two,:)=l_u_sample_new;
    %test the result value
    f_u_l_two =result_value_u_l_two( u_l,l_u,rand_trainset_two);
    
            %update l_p of trainset_three with first order
    l_p_two_sample=l_p(l_u_unique_two,:);
    opt_l_p_two = minFunc(@list_update_l_p_two,l_p_two_sample(:),options,p_l,trainset_two_sample,lamda,size(l_u_unique_two,1),matrix_feature);
    l_p_sample_new=reshape(opt_l_p_two, size(l_u_unique_two,1), matrix_feature);
    l_p(l_u_unique_two,:)=l_p_sample_new;
    %test the result value
    f_p_l_two =result_value_p_l_two( p_l,l_p,rand_trainset_two);
    
    %update u_l of trainset_two
    u_l_unique_two=unique(rand_trainset_two(:,1),'stable');
    u_l_two_sample=u_l(u_l_unique_two,:);
    trainset_two_sample=rand_trainset_two;
    u_l_two = containers.Map(u_l_unique_two(:,1),1:size(u_l_unique_two,1));
    userline_new=arrayfun(@(i) u_l_two(trainset_two_sample(i,1)),1:size(trainset_two_sample,1));
    trainset_two_sample(:,1)=userline_new';
    opt_u_l_two = minFunc(@list_update_u_l_two,u_l_two_sample(:),options,l_u,trainset_two_sample,lamda,size(u_l_unique_two,1),matrix_feature);
    u_l_sample_new=reshape(opt_u_l_two, size(u_l_unique_two,1), matrix_feature);
    u_l(u_l_unique_two,:)=u_l_sample_new;
    %test the result value
    f_u_l_two =result_value_u_l_two( u_l,l_u,rand_trainset_two);
    
             %update p_l of trainset_two
    p_l_unique_two=unique(rand_trainset_two(:,2),'stable');
    p_l_two_sample=p_l(p_l_unique_two,:);
    trainset_two_sample=rand_trainset_two;
    p_l_two = containers.Map(p_l_unique_two(:,1),1:size(p_l_unique_two,1));
    locationline_new=arrayfun(@(i) p_l_two(trainset_two_sample(i,2)),1:size(trainset_two_sample,1));
    trainset_two_sample(:,2)=locationline_new';
    opt_p_l_two = minFunc(@list_update_p_l_two,p_l_two_sample(:),options,l_p,trainset_two_sample,lamda,size(p_l_unique_two,1),matrix_feature);
    p_l_sample_new=reshape(opt_p_l_two, size(p_l_unique_two,1), matrix_feature);
    p_l(p_l_unique_two,:)=p_l_sample_new;
    %test the result value
    f_p_l_two =result_value_p_l_two( p_l,l_p,rand_trainset_two);
    
    end
    
    
    for i_one=1:iteration_one
        %trainset_one
    %give every train sample a negative sample. This negative sample must be a
    %location line number.
    rand_trainset_one_number=randsample(size(trainset_one,1),num_sample_one);
    rand_trainset_one=trainset_one(rand_trainset_one_number,:);
    [row_rand_trainset_one,col_rand_trainset_one]=size(rand_trainset_one);
    rand_negative_three=randsample(8,row_rand_trainset_one,'true');
    rand_trainset_one(:,4)=rand_negative_three;
    
%     %test the result value
    f_u_l_one =result_value_u_l_one( u_l,l_u,rand_trainset_one);
    f_p_l_one =result_value_p_l_one( p_l,l_p,rand_trainset_one);

    %update l_u of trainset_one with first order
    l_u_unique_one_first=unique(rand_trainset_one(:,3),'stable');
    l_u_unique_one_second=unique(rand_trainset_one(:,4),'stable');
    l_u_unique_one=unique([l_u_unique_one_first;l_u_unique_one_second],'stable');
    
    l_u_one_sample=l_u(l_u_unique_one,:);
    rand_trainset_one_sample=rand_trainset_one;
    l_u_one = containers.Map(l_u_unique_one(:,1),1:size(l_u_unique_one,1));
    
    locationline_new=arrayfun(@(i) l_u_one(rand_trainset_one_sample(i,3)),1:size(rand_trainset_one_sample,1));
    rand_trainset_one_sample(:,3)=locationline_new';
    locationline_new=arrayfun(@(i) l_u_one(rand_trainset_one_sample(i,4)),1:size(rand_trainset_one_sample,1));
    rand_trainset_one_sample(:,4)=locationline_new';
    opt_l_u_one = minFunc(@list_update_l_u_one,l_u_one_sample(:),options,u_l,rand_trainset_one_sample,lamda,size(l_u_unique_one,1),matrix_feature);
    l_u_sample_new=reshape(opt_l_u_one, size(l_u_unique_one,1), matrix_feature);
    l_u(l_u_unique_one,:)=l_u_sample_new;
    %test the result value
    f_u_l_one =result_value_u_l_one( u_l,l_u,rand_trainset_one);
    
            %update l_p of trainset_one with first order
    l_p_one_sample=l_p(l_u_unique_one,:);
    opt_l_p_one = minFunc(@list_update_l_p_one,l_p_one_sample(:),options,p_l,rand_trainset_one_sample,lamda,size(l_u_unique_one,1),matrix_feature);
    l_p_sample_new=reshape(opt_l_p_one, size(l_u_unique_one,1), matrix_feature);
    l_p(l_u_unique_one,:)=l_p_sample_new;
    %test the result value
    f_p_l_one =result_value_p_l_one( p_l,l_p,rand_trainset_one);
    
         %update u_l of trainset_one
    u_l_unique_one=unique(rand_trainset_one(:,1),'stable');
    u_l_one_sample=u_l(u_l_unique_one,:);
    rand_trainset_one_sample=rand_trainset_one;
    u_l_one = containers.Map(u_l_unique_one(:,1),1:size(u_l_unique_one,1));
    userline_new=arrayfun(@(i) u_l_one(rand_trainset_one_sample(i,1)),1:size(rand_trainset_one_sample,1));
    rand_trainset_one_sample(:,1)=userline_new';
    opt_u_l_one = minFunc(@list_update_u_l_one,u_l_one_sample(:),options,l_u,rand_trainset_one_sample,lamda,size(u_l_unique_one,1),matrix_feature);
    u_l_sample_new=reshape(opt_u_l_one, size(u_l_unique_one,1), matrix_feature);
    u_l(u_l_unique_one,:)=u_l_sample_new;
    %test the result value
    f_u_l_one =result_value_u_l_one( u_l,l_u,rand_trainset_one);
    
             %update p_l of trainset_one
    p_l_unique_one=unique(rand_trainset_one(:,2),'stable');
    p_l_one_sample=p_l(p_l_unique_one,:);
    rand_trainset_one_sample=rand_trainset_one;
    p_l_one = containers.Map(p_l_unique_one(:,1),1:size(p_l_unique_one,1));
    locationline_new=arrayfun(@(i) p_l_one(rand_trainset_one_sample(i,2)),1:size(rand_trainset_one_sample,1));
    rand_trainset_one_sample(:,2)=locationline_new';
    opt_p_l_one = minFunc(@list_update_p_l_one,p_l_one_sample(:),options,l_p,rand_trainset_one_sample,lamda,size(p_l_unique_one,1),matrix_feature);
    p_l_sample_new=reshape(opt_p_l_one, size(p_l_unique_one,1), matrix_feature);
    p_l(p_l_unique_one,:)=p_l_sample_new;
    %test the result value
    f_p_l_one =result_value_p_l_one( p_l,l_p,rand_trainset_one);
    
    end
    
%     if  0==mod(iteration_i,1)
        %prediction
%         [result_vector_1,result_vector_5,result_vector_10,result_vector_20,result_vector_30,result_vector_40,result_vector_50,result_vector_60,result_vector_70,result_vector_80,result_vector_90,result_vector_100]=arrayfun(@(i) list_prediction_Gowalla( testset,l_u,u_l,l_p,p_l,rho,location_unique_test,distance_matrix_frac,i),1:row_testset);
%         list_prediction_Gowalla( testset,l_u,u_l,l_p,p_l,rho,location_unique_test,distance_matrix_frac,1);

        [result_vector_1,result_vector_5,Myresult_location_1,Myresult_location_5,Myresult_location_10,Myresult_location_20,Myresult_location_30,Myresult_location_40,Myresult_location_50]=arrayfun(@(i) list_prediction_MLE( testset,l_u,u_l,l_p,p_l,i,distance_matrix_frac,firstCategoryLocation,location_unique_test,testset_location_category),1:row_testset);
        disp(['top1=£º' num2str(sum(result_vector_1)./ row_testset)]);
        disp(['top5=£º' num2str(sum(result_vector_5)./ row_testset)]);
%         disp(['top10=£º' num2str(sum(result_vector_10)./ row_testset)]);
%         disp(['top20=£º' num2str(sum(result_vector_20)./ row_testset)]);
%         disp(['top30=£º' num2str(sum(result_vector_30)./ row_testset)]);
%         disp(['top40=£º' num2str(sum(result_vector_40)./ row_testset)]);
%         disp(['top50=£º' num2str(sum(result_vector_50)./ row_testset)]);
%         disp(['top60=£º' num2str(sum(result_vector_60)./ row_testset)]);
        
        disp(['topLocation1=£º' num2str(sum(Myresult_location_1)./ row_testset)]);
        disp(['topLocation5=£º' num2str(sum(Myresult_location_5)./ row_testset)]);
        disp(['topLocation10=£º' num2str(sum(Myresult_location_10)./ row_testset)]);
        disp(['topLocation20=£º' num2str(sum(Myresult_location_20)./ row_testset)]);
        disp(['topLocation30=£º' num2str(sum(Myresult_location_30)./ row_testset)]);
        disp(['topLocation40=£º' num2str(sum(Myresult_location_40)./ row_testset)]);
        disp(['topLocation50=£º' num2str(sum(Myresult_location_50)./ row_testset)]);
%         disp(['top70=£º' num2str(sum(result_vector_70)./ row_testset)]);
%         disp(['top80=£º' num2str(sum(result_vector_80)./ row_testset)]);
%         disp(['top90=£º' num2str(sum(result_vector_90)./ row_testset)]);
%         disp(['top100=£º' num2str(sum(result_vector_100)./ row_testset)]);
        disp(['iteration£º' num2str(iteration_i)]);
%         disp(['rho£º' num2str(rho)]);

        fprintf(fid,'top1=£º%s\n',num2str(sum(result_vector_1)./ row_testset));
        fprintf(fid,'top5=£º%s\n',num2str(sum(result_vector_5)./ row_testset));
%         fprintf(fid,'top10=£º%s\n',num2str(sum(result_vector_10)./ row_testset));
%         fprintf(fid,'top20=£º%s\n',num2str(sum(result_vector_20)./ row_testset));
%         fprintf(fid,'top30=£º%s\n',num2str(sum(result_vector_30)./ row_testset));
%         fprintf(fid,'top40=£º%s\n',num2str(sum(result_vector_40)./ row_testset));
%         fprintf(fid,'top50=£º%s\n',num2str(sum(result_vector_50)./ row_testset));
%         fprintf(fid,'top60=£º%s\n',num2str(sum(result_vector_60)./ row_testset));
        
        fprintf(fid,'topLocation1=£º%s\n',num2str(sum(Myresult_location_1)./ row_testset));
        fprintf(fid,'topLocation5=£º%s\n',num2str(sum(Myresult_location_5)./ row_testset));
        fprintf(fid,'topLocation10=£º%s\n',num2str(sum(Myresult_location_10)./ row_testset));
        fprintf(fid,'topLocation20=£º%s\n',num2str(sum(Myresult_location_20)./ row_testset));
        fprintf(fid,'topLocation30=£º%s\n',num2str(sum(Myresult_location_30)./ row_testset));
        fprintf(fid,'topLocation40=£º%s\n',num2str(sum(Myresult_location_40)./ row_testset));
        fprintf(fid,'topLocation50=£º%s\n',num2str(sum(Myresult_location_50)./ row_testset));
        fprintf(fid,'iteration£º%s\n',num2str(iteration_i));
        fprintf(fid,'\n');
%     end %end if  0==mod(iteration_i,50)
    

    
    
    
end % end while

fclose(fid);