function [best_ann, best_R2] = rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, y_test_or, mean_ytest, sd_ytest, lag)
%[best_ann, best_R2] = rep_ann(M_ann_cal,Y_ann_cal, M_ann_val, y_test_or, mean_ytest, sd_ytest, lag)
%
%   Detailed explanation goes here
max_rep=2;
max_neurons=10;
jump_neurons=5;
max_levels=2;
best_R2=0;
for levels=1:max_levels
    for neurons=jump_neurons:jump_neurons:max_neurons
        for rep=1:max_rep
            neuron_vet=ones(1,levels)*neurons;
            ann_temp=feedforwardnet(neuron_vet);
            ann_temp=train(ann_temp, M_ann_cal', Y_ann_cal');
            y_val_hat =ann_temp(M_ann_val')';
            y_val_hat =y_val_hat.*sd_ytest(lag+3:end)+mean_ytest(lag+3:end);
            sq_err=sum((y_val_hat-y_test_or(lag+3:end)).^2);
            R2_temp=1-sq_err/sum((y_test_or(lag+3:end)-mean_ytest(lag+3:end)).^2);
            if (R2_temp>best_R2)
                best_R2=R2_temp;
                best_ann=ann_temp;
            end
        end
    end
end
end