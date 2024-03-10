#include <iostream>
#include "DataVector.h"
using namespace std;

// Constructor
DataVector::DataVector(int dimension)
{
    setDimension(dimension);
}

// Destructor
DataVector::~DataVector()
{
}

// Copy Constructor
DataVector::DataVector(const DataVector &other) : v(other.v)
{
}

DataVector &DataVector::operator=(const DataVector &other)
{
    if (this != &other)
    {
        v = other.v;
    }
    return *this;
}

void DataVector::setDimension(int dimension)
{
    // Clear existing data
    v.clear();
    // Resize the vector to the new dimension and initialize with zeros
    v.resize(dimension, 0.0);
}


DataVector DataVector::operator+(const DataVector &other) const
{
    DataVector result(*this);
    for (size_t i = 0; i < v.size(); ++i)
    {
        result.v[i] += other.v[i];
    }
    return result;
}



// Overloaded operators for vector subtraction
DataVector DataVector::operator-(const DataVector &other) const
{
    DataVector result(*this); // Create a copy of the current vector
    for (size_t i = 0; i < v.size(); ++i)
    {
        result.v[i] -= other.v[i];
    }
    return result;
}

// Overloaded operator for dot product
double DataVector::operator*(const DataVector &other) const
{
    double result = 0.0;
    for (size_t i = 0; i < v.size(); ++i)
    {
        result += v[i] * other.v[i];
    }
    return result;
}

vector<double> DataVector:: getVector() const{
    return v;
}

// Member function to calculate the norm (length) of the vector
double DataVector::norm() const
{
    return sqrt(*this * *this);
}

// Member function to calculate the distance between two vectors
double DataVector::dist(const DataVector &other) const
{
    DataVector diff = *this - other;
    return diff.norm();
}

void DataVector::setValues(const std::vector<double> &values)
{
    v = values;
}

size_t DataVector::getDimension() const
{
    return v.size();
}

// Read only. Access value at an index
double DataVector::operator[](size_t index) const
{
    if (index < v.size())
    {
        return v[index];
    }
    else
    {
        // Handle error: Index out of bounds
        return 0.0;
    }
}

// Write.

double &DataVector::operator[](size_t index)
{
    if (index < v.size())
    {
        return v[index];
    }
    else
    {
        // Handle error: Index out of bounds
        return v[0];
    }
}
