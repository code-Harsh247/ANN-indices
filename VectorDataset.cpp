#include "VectorDataset.h"
#include <fstream>
#include <algorithm>
using namespace std;

// Default constructor
VectorDataset::VectorDataset() {}

// Parameterized constructor with initial dataset
VectorDataset::VectorDataset(const std::vector<DataVector>& data) : dataset(data) {}

// Destructor
VectorDataset::~VectorDataset() {}

// Getter for the dataset
const vector<DataVector>& VectorDataset::getDataset() const{
    return dataset;
}
void VectorDataset:: pushValue(const DataVector& data){
    dataset.push_back(data);
}
size_t VectorDataset::getDimension() const
{
    return dataset.size();
}

DataVector VectorDataset:: getDataVector(int index) const{
    return dataset[index];
}
// Setter for the dataset
void VectorDataset::setDataset(const vector<DataVector>& data) {
    dataset = data;
}

// Read dataset from a CSV file
void VectorDataset::readDataset(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Could not read file!" << endl;
        return;
    }

    vector<DataVector> data;
    string line;

    // Read each line from the file
    while (getline(file, line)) {
        istringstream iss(line);
        double value;
        vector <double> vec;

        // Parse values separated by commas
        while (iss >> value) {
            vec.push_back(value);
            if (iss.peek() == ',')
                iss.ignore(); // Ignore the comma
        }

        // Create a DataVector object and set its values
        DataVector vector(vec.size());
        vector.setValues(vec);
        data.push_back(vector);
    }

    file.close();

    // Set the read dataset
    setDataset(data);
}

// k-Nearest Neighbor search function
VectorDataset VectorDataset::kNearestNeighbor(const DataVector& testVector, size_t k) const {
    VectorDataset resultDataset(*this); // Create a copy of the dataset
    cout << "Nearest N running" << endl;

    // Sort the dataset based on the distance to the testVector
    auto distanceComparator = [&testVector](const DataVector& a, const DataVector& b) {
        return a.dist(testVector) < b.dist(testVector);
    };

    sort(resultDataset.dataset.begin(), resultDataset.dataset.end(), distanceComparator);

    cout << "Sorting done" << endl;

    // Keep only the top k vectors
    if (k < resultDataset.dataset.size()) {
        resultDataset.dataset.resize(k);
    }

    return resultDataset;
}
