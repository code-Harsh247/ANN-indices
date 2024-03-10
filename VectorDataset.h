/*

The VectorDataset class represents a dataset of DataVector objects. It provides
functionality to read a dataset from a file, perform k-Nearest Neighbor searches,
and store the dataset for further analysis.
The class is designed to work seamlessly with the DataVector class, allowing users
to create and manage datasets of multi-dimensional vectors easily. It abstracts away
the complexity of dataset handling, offering a clean interface for common operations.

Constructors:
1) VectorDataset(): Constructs an empty VectorDataset.
2) VectorDataset(const std::vector<DataVector>& data): Constructs a VectorDataset with the provided dataset.
3) ~VectorDataset(): Destructs the VectorDataset and cleans up associated resources.

Data Access Functions
1) const std::vector<DataVector>& getDataset() const: Gets the dataset.
2) void setDataset(const std::vector<DataVector>& data): Sets the dataset.

File I/O Functions
1) void readDataset(const std::string& filename): Reads the dataset from a file.

k-Nearest Neighbor Function
VectorDataset kNearestNeighbor(const DataVector& testVector, size_t k) const: Performs k-Nearest Neighbor search 
on the dataset. Sorts the dataset based on difference with the test vector and then selects only the first k ones. 

*/


#ifndef VECTORDATASET_H
#define VECTORDATASET_H

#include "DataVector.h"
#include <vector>
using namespace std;

class VectorDataset {
private:
    vector<DataVector> dataset;

public:
    // Constructors and destructor
    VectorDataset();
    VectorDataset(const std::vector<DataVector>& data);
    ~VectorDataset();

    // Data access functions
    const vector<DataVector>& getDataset() const;
    void setDataset(const vector<DataVector>& data);
    void pushValue(const DataVector& value);
    size_t getDimension() const; 

    // Read dataset from file
    void readDataset(const string& filename);

    DataVector getDataVector(int index) const;
    // k-Nearest Neighbor function
    VectorDataset kNearestNeighbor(const DataVector& testVector, size_t k) const;
};

#endif  // VECTORDATASET_H
