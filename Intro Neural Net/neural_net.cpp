#include <stdlib.h>
#include <iostream>
#include <math.h>

//compile with g++ neural_net.cpp -o neural_net -lm

using namespace std;

//ideally this should be stored in a file but for these purposes making it a global variable is ideal
float synaptic_weights[3]; //the value of the trained weights

//calculates the sigmoid function on the input
float sigmoid(float input){
    return 1/(1 + exp(-input));
}

//calculates the derivative of the sigmoid function on the input
float sigmoid_deriv(float input){
    return input*(1 - input);
}

//trains the neural network
void train(){
    //the first to elements of each internal array are the inputs, the third is the bias
    int training_inputs[4][3] = {{1,1,1}, {1,0,1}, {0,1,1}, {0,0,1}};
    
    //expected outputs from the neural network
    //int training_outputs[4] = {0,0,0,1}; //NOR gate
    int training_outputs[4] = {1,1,1,0}; //OR gate
    //int training_outputs[4] = {0,1,1,0}; //XOR gate (can't do)
    //int training_outputs[4] = {1,0,0,0}; //AND gate
    //int training_outputs[4] = {0,1,1,1}; //NAND gate
    //int training_outputs[4] = {1,0,0,1}; //XNOR gate (can't do)
    
    //seed the weights for both the inputs and the bias
    srand(time(NULL));
    synaptic_weights[0] = (float) rand()/RAND_MAX;
    synaptic_weights[1] = (float) rand()/RAND_MAX;
    synaptic_weights[2] = (float) rand()/RAND_MAX;
    float outputs[4];
    //train the neural network 20000 times
    for (int i = 0; i < 20000; i++){
        for (int j = 0; j < 4; j++){
            //sigmoid of the dot product of the inputs with the weights to get the ouputs for each input
            outputs[j] = sigmoid(training_inputs[j][0]*synaptic_weights[0] + training_inputs[j][1]*synaptic_weights[1] + training_inputs[j][2]*synaptic_weights[2]);
        }
        float error[4];
        float adjustments[4];
        for (int j = 0; j < 4; j++){
            //subtract calculated output from expected output to get the error
            error[j] = training_outputs[j] - outputs[j];
            //calculate adjustments for each weight based of the sigmoid derivative of each output
            adjustments[j] = error[j] * sigmoid_deriv(outputs[j]);
        }
        //take the dot product of the training inputs with the calculated adjustments and add that to the relevant weights
        for (int j = 0; j < 3; j++){
            synaptic_weights[j] += training_inputs[0][j]*adjustments[0] + training_inputs[1][j]*adjustments[1] + training_inputs[2][j]*adjustments[2] + training_inputs[3][j]*adjustments[3];
        }
    }
    /*for (int i = 0; i < 4; i++){
        cout << outputs[i] << endl;
    }*/
}

//returns an output to the given inputs
float answer(int input1, int input2){
    //bias is assumed to be 1
    int bias = 1;
    //calculates the sigmoid of the dot product of the weights with the inputs and bias and rounds it to achieve a 1 or a 0
    return round(sigmoid(synaptic_weights[0]*input1 + synaptic_weights[1]*input2 + synaptic_weights[2]*bias));
}

int main(){
    //train the algorithm
    train();
    int input1;
    int input2;
    cout << "Input 1: ";
    cin >> input1;
    cout << "\nInput 2: ";
    cin >> input2;
    cout << "\n" << answer(input1, input2) << endl;
}