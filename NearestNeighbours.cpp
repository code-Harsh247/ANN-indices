#include <bits/stdc++.h>
#include "TreeIndex.h"
using namespace std;

int main()
{
    // Create a VectorDataset object and read the dataset from a CSV file
    VectorDataset dataset;
    string trainData;
    cout << "Enter the CSV file name for the vector datasheet: " << endl;
    cin >> trainData;

    // Check if the file exists
    ifstream fileCheck(trainData);
    if (!fileCheck)
    {
        cout << "Error: The file " << trainData << " does not exist." << endl;
        return 1; // Exit with an error code
    }

    // Read the dataset from the CSV file
    dataset.readDataset(trainData);

    // Get the test vector from the user
    vector<double> testVectorValues;
    VectorDataset testVectorSet;
    string testVectorFileName;

    cout << "Enter the CSV file name for the test vector: ";
    cin >> testVectorFileName;

    // Open the test vector file
    ifstream testVectorFile(testVectorFileName);

    if (!testVectorFile.is_open())
    {
        cout << "Could not open the test vector file: " << testVectorFileName << endl;
        return 1;
    }

    VectorDataset testDataset;
    testDataset.readDataset(testVectorFileName);

    int n;
    cout << "Enter 1 for KDTree and 2 for RPTree: ";
    cin >> n;

    string nearestNeighborsFileName = "nearest_neighbors.csv";
    ofstream nearestNeighborsFile(nearestNeighborsFileName);

    // 1 for KDTree 2 for RPTree
    if (n == 1)
    {
        KDTreeIndex KDTREE = KDTreeIndex::GetInstance();
        for (int i = 0; i < dataset.getDataset().size(); i++)
        {
            KDTREE.AddData(dataset.getDataVector(i));
        }
        KDTreeIndex *tree = KDTREE.MakeTree();
        cout << "Tree constructed" << endl;
        size_t k;
        cout << "Enter the value of k: ";
        cin >> k;
        vector<DataVector> testVectors = testDataset.getDataset();
        for (auto testVector : testVectors)
        {
            VectorDataset kNearestNeighbors = KDTREE.Search(testVector, k);
            const vector<DataVector> &neighbors = kNearestNeighbors.getDataset();

            if (!nearestNeighborsFile.is_open())
            {
                cout << "Could not create the nearest neighbors file: " << nearestNeighborsFileName << endl;
                return 1;
            }

            // Write the nearest neighbors to the CSV file
            for (size_t i = 0; i < neighbors.size(); ++i)
            {
                const DataVector &neighbor = neighbors[i];
                size_t dimension = neighbor.getDimension();

                for (size_t j = 0; j < dimension; ++j)
                {
                    nearestNeighborsFile << neighbor[j];
                    if (j < dimension - 1)
                    {
                        nearestNeighborsFile << ",";
                    }
                }

                nearestNeighborsFile << "\n";
            }
            nearestNeighborsFile << "\n";
        }
    }
    else
    {
        cout << "Invalid input" << endl;
    }
}