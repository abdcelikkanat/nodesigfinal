#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <random>
#include <sstream>
#include <fstream>
#include <string>
#include <bitset>
#include <chrono>
#include "Graph.h"
#include <omp.h>

using namespace std;


template<typename T>
class Model {
private:
    random_device _rd;
    int _headerBlockSize = 4; // in bytes

    unsigned int _dim;
    unsigned int _numOfNodes;
    T **_weights;
    string _weightDistr;
    bool _cyclicweights;
    bool _verbose;

    void _generateWeights(unsigned int N, unsigned int M, unsigned int walklen);

public:

    Model(unsigned int numOfNodes, unsigned int dim, string weightDistr, bool cyclicweights, bool verbose);
    ~Model();
    void learnEmb(vector <vector <pair<unsigned int, T>>> P, unsigned int walkLen, T alpha, string filePath);

};

template<typename T>
Model<T>::Model(unsigned int numOfNodes, unsigned int dim, string weightDistr, bool cyclicweights, bool verbose) {

    if(dim % 8 != 0) {
        cout << "The embedding dimension must be divisible by 8." << endl;
        throw;
    }

    this->_dim = dim;
    this->_numOfNodes = numOfNodes;
    this->_weightDistr = weightDistr;
    this->_cyclicweights = cyclicweights;
    this->_verbose = verbose;

}

template<typename T>
Model<T>::~Model() {

    for(unsigned int n=0; n<this->_numOfNodes; n++)
        delete [] _weights[n];
    delete [] _weights;

}

template<typename T>
void Model<T>::_generateWeights(unsigned int N, unsigned int M, unsigned int walklen) {

    if (this->_verbose)
        cout << "\t- A weight matrix of size " << this->_numOfNodes << "x" << this->_dim << " is being (re)generated." << endl;

    default_random_engine generator(this->_rd());

    bool cauchy_distr;

    if( this->_weightDistr.compare("cauchy") == 0 ) {

        cauchy_distribution<T> distribution(0.0, 1.0);

        if( this->_cyclicweights ) {

            this->_weights = new T *[N];
            for (unsigned int n = 0; n < N; n++) {
                this->_weights[n] = new T[M];
                if(n % M == 0) {
                    for (unsigned int nb = 0; nb < M; nb++) {
                        this->_weights[n][nb] = distribution(generator) / walklen;
                    }
                } else {
                    for (unsigned int nb = 0; nb < M; nb++)
                        this->_weights[n][nb] = this->_weights[n-1][(nb+1)%M];
                }
            }

        } else {

            this->_weights = new T *[N];
#           pragma omp parallel for
            for (unsigned int n = 0; n < N; n++) {
                this->_weights[n] = new T[M];
                for (unsigned int nb = 0; nb < M; nb++) {
                    this->_weights[n][nb] = distribution(generator) / walklen;
                }
            }

        }

    }

    if( this->_weightDistr.compare("gauss") == 0 ) {

        normal_distribution<T> distribution(0.0, 1.0);

        if( this->_cyclicweights ) {

            this->_weights = new T *[N];
            for (unsigned int n = 0; n < N; n++) {
                this->_weights[n] = new T[M];
                if(n % M == 0) {
                    for (unsigned int nb = 0; nb < M; nb++) {
                        this->_weights[n][nb] = distribution(generator);
                    }
                } else {
                    for (unsigned int nb = 0; nb < M; nb++)
                        this->_weights[n][nb] = this->_weights[n-1][(nb+1)%M];
                }
            }

        } else {

            this->_weights = new T *[N];
#           pragma omp parallel for
            for (unsigned int n = 0; n < N; n++) {
                this->_weights[n] = new T[M];
                for (unsigned int nb = 0; nb < M; nb++) {
                    this->_weights[n][nb] = distribution(generator);
                }
            }

        }

    }

    if (this->_verbose)
        cout << "\t- Completed!" << endl;

}

template<typename T>
void Model<T>::learnEmb(vector <vector <pair<unsigned int, T>>> P, unsigned int walkLen, T alpha, string filePath) {

    this->_generateWeights(this->_numOfNodes, this->_dim, walkLen);


    T **current = new T*[this->_numOfNodes];
    for(unsigned int node=0; node < this->_numOfNodes; node++)
        current[node] = new T[this->_dim]{0};

    if(this->_verbose)
        cout << "+ The computation of walks has just started." << endl;

    for(int l=0; l<walkLen; l++) {

        if(this->_verbose)
            cout << "\t- Walk: " << l+1 << "/" << walkLen << endl;

        T **temp = new T*[this->_numOfNodes];
        for(unsigned int node=0; node < this->_numOfNodes; node++)
            temp[node] = new T[this->_dim]{0};

        #pragma omp parallel for
        for (unsigned node = 0; node < this->_numOfNodes; node++) {

            for (int d = 0; d < this->_dim; d++) {
                temp[node][d] = 0;
                for (unsigned int nbIdx = 0; nbIdx < P[node].size(); nbIdx++)
                    temp[node][d] += (current[get<0>(P[node][nbIdx])][d] + this->_weights[get<0>(P[node][nbIdx])][d]) *
                                     get<1>(P[node][nbIdx]);

                temp[node][d] *= alpha;
            }

        }

        for(unsigned int node=0; node < this->_numOfNodes; node++)
            delete[] current[node];
        delete [] current;
        current = temp;


    }


    fstream fs(filePath, fstream::out | fstream::binary);
    if(fs.is_open()) {

        // Write the header
        fs.write(reinterpret_cast<const char *>(&_numOfNodes), _headerBlockSize);
        fs.write(reinterpret_cast<const char *>(&_dim), _headerBlockSize);

        for(unsigned int node=0; node< this->_numOfNodes; node++) {

            vector<uint8_t> bin(_dim/8, 0);
            for (unsigned int d = 0; d < _dim; d++) {
                bin[int(d/8)] <<= 1;
                if (current[node][d] > 0)
                    bin[int(d/8)] += 1;
            }
            copy(bin.begin(), bin.end(), std::ostreambuf_iterator<char>(fs));
        }

        fs.close();

    } else {
        cout << "+ An error occurred during opening the file!" << endl;
    }

    for(unsigned int node=0; node < this->_numOfNodes; node++)
        delete[] current[node];
    delete [] current;


}










#endif //MODEL_H