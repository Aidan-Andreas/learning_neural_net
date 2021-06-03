//This project is created following guidelines set forth by Rising Odegua in the article "Building a Neural Network From Scratch Using Python (Part 1)" published April 1, 2020 at heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432
//The file heart.dat is provided by the UCI Machine Learning Repository archive.ics.uci.edu/ml/datasets/Statlog+%28Heart%29
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <bits/stdc++.h>

//compile with g++ neural_net.cpp -o neural_net -lm

using namespace std;

//where the training data from heart.dat will be stored
float training_input_data[270][13];
float training_output_data[270];
int data_set_size = 270;
int hidden_layers = 8;
int inputs = 13;
float learning_rate = 0.001;
float iterations = 1000;
float w1[8][13];//input layer weights 13x8 array (created with the rows and columns flipped)
float w2[8];//hidden layer weights 8x1 array
float b1[8]; //input layer bias 8x1 array
float b2; //hidden layer bias 1x1 array

//calculates the ReLu (Rectified Linear Unit) activation function on the input
float relu(float val){
    if (val < 0) val = 0;
    return val;
}

//derivative of relu
float dRelu(float val){
    if (val > 0) val = 1;
    if (val <= 0) val = 0;
    return val;
}

//calculates the sigmoid function on the input
float sigmoid(float input){
    return 1/(1 + exp(-input));
}

//takes the dot product of the two arrays of length len
float dot_product(float arr1[], float arr2[], int len){
    float sum = 0;
    for (int i = 0; i < len; i++){
        sum += arr1[i] * arr2[i];
    }
    return sum;
}

//prevents issues where value being 0 or less than 0 causes an error by setting it to 0.00000001
float eta(float val){
    if (val <= 0) val = 0.00000001;
    return val;
}

void train(){
    //fill both layer's weights with random numbers between 0 and 1
    srand(time(NULL));
    for (int i = 0; i < inputs; i++){
        for (int j = 0; j < 8; j++){
            w1[j][i] = (float) rand()/RAND_MAX;
            w1[j][i] = w1[j][i]/2;
            cout << w1[j][i] << " ";
        }
        cout << endl;
    }
    for (int i = 0; i < hidden_layers; i++){
        w2[i] = (float) rand()/RAND_MAX;
        w2[i] = w2[i]/2;
        b1[i] = (float) rand()/RAND_MAX;
        b1[i] = b1[i]/2;
    }
    b2 = (float) rand()/RAND_MAX;
    b2 = b2/2;

    for (int iter = 0; iter < iterations; iter++){
        for (int data_set_ind = 0; data_set_ind < data_set_size; data_set_ind++){
            //forward propagation
            float A1[hidden_layers];
            float Z1[hidden_layers];
            for (int j = 0; j < hidden_layers; j++){
                Z1[j] = dot_product(training_input_data[data_set_ind], w1[j], 13) + b1[j];
                A1[j] = relu(Z1[j]);
            }
            float Z2 = dot_product(A1, w2, hidden_layers) + b2;
            float pred_output = sigmoid(Z2);


            //backward propagation
            float real_output = training_output_data[data_set_ind];
            float real_output_inv = 1 - real_output;
            float pred_output_inv = 1 - pred_output;
            
            //I'm shorthanding the derivative naming of the following variable by putting a d in front
            float dpred_output = real_output_inv/eta(pred_output_inv) - real_output/eta(pred_output);
            float dsigmoid = pred_output * pred_output_inv;

            float dw1[8][13];
            float dw2[8];
            float db1[8];
            float db2;
            float dz1[8];
            float dz2;
            float da1[8];

            dz2 = dpred_output * dsigmoid;
            if (dsigmoid == 0) cout << "dsig was 0" << endl;
            db2 = dz2;
            
            for (int i = 0; i < hidden_layers; i++){
                da1[i] = dz2 * w2[i];
                dw2[i] = A1[i] * dz2;
                dz1[i] = da1[i] * dRelu(Z1[i]);
                db1[i] = dz1[i];
            }

            for (int i = 0; i < hidden_layers; i++){
                for (int j = 0; j < inputs; j++){
                    dw1[i][j] = training_input_data[data_set_ind][j]*dz1[i];
                }
            }

            for (int i = 0; i < hidden_layers; i++){
                for (int j = 0; j < inputs; j++){
                    w1[i][j] = w1[i][j] - (learning_rate * dw1[i][j]);
                }
            }

            for (int i = 0; i < hidden_layers; i++){
                w2[i] = w2[i] - (learning_rate * dw2[i]);
                b1[i] = b1[i] - (learning_rate * db1[i]);
            }

            b2 = b2 - (learning_rate * db2);


            /*float dz2 = dpred_output * dsigmoid;
            float da1 = 0;
            for (int i = 0; i < hidden_layers; i++){
                da1 += dz2 * w2[i];
            }

            float dw2 = 0;
            float dz1[hidden_layers];
            float db1[hidden_layers];
            for (int i = 0; i < hidden_layers; i++){
                dw2 += dz2 * A1[i];
                dz1[i] = da1 * dRelu(Z1[i]);
                db1[i] = dz1[i];
            }
            float db2 = dz2;
            
            float dw1[13];
            for (int i = 0; i < 13; i++){
                for (int j = 0; j < hidden_layers; j++){
                    dw1[i] += training_input_data[data_set_ind][i]*dz1[j];
                }
            }

            for (int i = 0; i < hidden_layers; i++){
                for (int j = 0; j < 13; j++){
                    w1[i][j] = w1[i][j] - (learning_rate * dw1[j]);
                }
            }

            for (int i = 0; i < hidden_layers; i++){
                w2[i] = w2[i] - (learning_rate * dw2);
                b1[i] = b1[i] - (learning_rate * db1[i]);
            }

            b2 = b2 - (learning_rate * db2);*/
        }
    }
}

float predict(int index){
    float A1[hidden_layers];
    float Z1[hidden_layers];
    for (int j = 0; j < hidden_layers; j++){
        Z1[j] = dot_product(training_input_data[index], w1[j], inputs) + b1[j];
        A1[j] = relu(Z1[j]);
    }
    float Z2 = dot_product(A1, w2, hidden_layers) + b2;
    float pred_output = sigmoid(Z2);
    return pred_output;
}

int main(){

    string line;
    ifstream file("heart.dat");
    int obs = 0;//observation number

    float minArr[13]; //used to standardize the data between 0 and 1 contains the minimum value of each feature
    for (int i = 0; i < inputs; i++){
        minArr[i] = MAXFLOAT;
    }
    float maxArr[13]; //used to standardize the data between 0 and 1 contains the maximum value of each feature (none of the features are < 0)

    //still need to create a test set from this data at some point
    while (getline(file, line)){
        stringstream ss(line);
        for (int feature = 0; feature < inputs; feature++){ //ignoring the heart disease feature since that is the outcome
            string val;
            ss >> val;
            training_input_data[obs][feature] = stof(val);//converts the value to a float and stores it in the training data array;
            if (training_input_data[obs][feature] < minArr[feature]) minArr[feature] = training_input_data[obs][feature];
            if (training_input_data[obs][feature] > maxArr[feature]) maxArr[feature] = training_input_data[obs][feature];
        }

        string val;
        ss >> val;
        training_output_data[obs] = stof(val) - 1; //heart disease is a 1 or a 2 in this data set, this makes it a 0 or a 1 and stores it in the training output data array
        obs++;
    }

    //normalize the data by subtracting the minimum value of a feature from every feature of that type and dividing the maximum value of a feature of that type (which will be the max before subtraction minus the min)
    for (int data_set_ind = 0; data_set_ind < data_set_size; data_set_ind++){
        for (int feature = 0; feature < inputs; feature++){
            training_input_data[data_set_ind][feature] = (training_input_data[data_set_ind][feature] - minArr[feature])/(maxArr[feature] - minArr[feature]);
        }
    }

    train();

    cout << endl;
    for (int i = 0; i < 13; i++){
        for (int j = 0; j < hidden_layers; j++) cout << w1[j][i] << " "; 
        cout << endl;
    }

    for (int i = 0; i < 10; i++){
        cout << predict(i) << endl;
    }
}