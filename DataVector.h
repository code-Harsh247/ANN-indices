/*

The DataVector class represents a vector of double values, providing various
operations for vector manipulation, mathematical computations, and distance
calculations. It is a fundamental building block for creating datasets and
performing operations in multi-dimensional space.


Constructors:
DataVector(int dimension = 0): Constructs a DataVector with the specified dimension.

Destructor:
~DataVector(): Destructs the DataVector and cleans up associated resources.

Member Functions:
1) void setDimension(int dimension = 0): Sets the dimension of the vector.
2) DataVector operator+(const DataVector& other) const: Adds two DataVector objects.
3) DataVector operator-(const DataVector& other) const: Subtracts another DataVector.
4) double operator*(const DataVector& other) const: Calculates the dot product.
5) double norm() const: Calculates the Euclidean norm.
6) double dist(const DataVector& other) const: Calculates the Euclidean distance.
7) void setValues(const std::vector<double>& values): Sets values for the vector.
8) void display() const: Displays the vector values.
9) size_t getDimension() const: Gets the dimension of the vector.
10) double operator[](size_t index) const: Accessor for individual elements.
11) double& operator[](size_t index): Mutable accessor for individual elements.

*/




#ifndef DATAVECTOR_H
#define DATAVECTOR_H

#include <bits/stdc++.h>
using namespace std;

class DataVector {
    private: 
        vector<double> v; 
    public: 
        DataVector(int dimension=0); 
        ~DataVector(); 
        DataVector(const DataVector& other); 
        DataVector & operator=(const DataVector &other); 
        void setDimension(int dimension=0); 
        DataVector operator+(const DataVector &other) const; 
        DataVector operator-(const DataVector &other) const; 
        double operator*(const DataVector &other) const; 
        double norm() const;
        double dist(const DataVector& other) const;

        // Function to set values for the vector
        void setValues(const vector<double>& values);
        vector<double> getVector() const; 
        void display() const;
        // Accessor for vector size
        size_t getDimension() const;
        // Additional member functions for accessing individual elements
        double operator[](size_t index) const;
        double& operator[](size_t index);
};

#endif  // DATAVECTOR_H