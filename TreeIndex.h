#ifndef TREEINDEX_H
#define TREEINDEX_H

#include "DataVector.h"
#include "VectorDataset.h"
using namespace std;

class TreeIndex
{
protected:
    VectorDataset Data;
    TreeIndex();

public:
    static TreeIndex &GetInstance();
    void AddData(const DataVector &dataVector);
    void RemoveData(const DataVector &dataVector);
    ~TreeIndex();
};

class KDTreeIndex : public TreeIndex
{
public:
    ~KDTreeIndex();
    static KDTreeIndex &GetInstance();
    KDTreeIndex *MakeTree();
    pair<VectorDataset,VectorDataset> ChooseRule();
    vector<DataVector> Search(const DataVector &query, int k);
    int maxSpreadAxis();
    double getMedian() const;
    int getAxis() const;
    double distanceToHyperplane(const DataVector& query, const KDTreeIndex* node);

private:
    KDTreeIndex();
    double median;
    int axis;
    KDTreeIndex *left;
    KDTreeIndex *right;
};

class RPTreeIndex : public TreeIndex {
private:
    struct Node {
        std::vector<DataVector> v;
        Node* left;
        Node* right;
        double medianval;
        std::vector<double> axis;
    };

    Node* root;

    // Private helper functions...
    std::vector<double> randomUnitDirection(size_t dimensions);
    int randomX(int k);
    Node* buildTree(std::vector<DataVector>& points);
    Node* buildTree(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
    std::vector<DataVector> search(DataVector &point, Node* node, int k);
    
    static RPTreeIndex *instance;

public:
    RPTreeIndex();
    static RPTreeIndex *GetInstance();
    void maketree(std::vector<DataVector> &points);
    std::vector<DataVector> query_search(DataVector &point, int k);
};

#endif