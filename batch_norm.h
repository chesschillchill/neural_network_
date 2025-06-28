#pragma once  
#include <iostream>  
#include <vector>  
#include <cmath>  
#include <numeric>  

class BatchNorm {  
private:  
   float epsilon = 1e-5f; // Hằng số để tránh chia cho 0  
   float momentum = 0.9f; // Momentum cho running mean và variance  
   std::vector<float> gamma; // Tham số scale  
   std::vector<float> beta; // Tham số shift  
   std::vector<float> running_mean; // Trung bình chạy  
   std::vector<float> running_var; // Phương sai chạy  
   bool training; // Chế độ huấn luyện hay suy luận  

public:  
   BatchNorm(int size, bool train = true) : training(train) {  
       gamma = std::vector<float>(size, 1.0f); // Khởi tạo gamma = 1  
       beta = std::vector<float>(size, 0.0f);  // Khởi tạo beta = 0  
       running_mean = std::vector<float>(size, 0.0f);  
       running_var = std::vector<float>(size, 1.0f);  
   }  

   std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input) {  
       int batch_size = input.size();  
       int feature_size = input[0].size();  
       std::vector<std::vector<float>> output(batch_size, std::vector<float>(feature_size));  

       if (training) {  
           // Tính trung bình và phương sai cho mỗi đặc trưng  
           std::vector<float> mean(feature_size, 0.0f);  
           std::vector<float> variance(feature_size, 0.0f);  

           // Tính trung bình  
           for (int j = 0; j < feature_size; ++j) {  
               for (int i = 0; i < batch_size; ++i) {  
                   mean[j] += input[i][j];  
               }  
               mean[j] /= batch_size;  
           }  

           // Tính phương sai  
           for (int j = 0; j < feature_size; ++j) {  
               for (int i = 0; i < batch_size; ++i) {  
                   variance[j] += std::pow(input[i][j] - mean[j], 2);  
               }  
               variance[j] /= batch_size;  
           }  

           // Chuẩn hóa và scale/shift  
           for (int i = 0; i < batch_size; ++i) {  
               for (int j = 0; j < feature_size; ++j) {  
                   float normalized = (input[i][j] - mean[j]) / std::sqrt(variance[j] + epsilon);  
                   output[i][j] = gamma[j] * normalized + beta[j];  
               }  
           }  

           // Cập nhật running mean và running variance  
           for (int j = 0; j < feature_size; ++j) {  
               running_mean[j] = momentum * running_mean[j] + (1.0f - momentum) * mean[j];  
               running_var[j] = momentum * running_var[j] + (1.0f - momentum) * variance[j];  
           }  
       }  
       else {  
           // Chế độ suy luận: sử dụng running mean và running variance  
           for (int i = 0; i < batch_size; ++i) {  
               for (int j = 0; j < feature_size; ++j) {  
                   float normalized = (input[i][j] - running_mean[j]) / std::sqrt(running_var[j] + epsilon);  
                   output[i][j] = gamma[j] * normalized + beta[j];  
               }  
           }  
       }  

       return output;  
   }  
};
