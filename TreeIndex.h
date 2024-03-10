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

class RPTreeIndex : public TreeIndex
{
public:
    static RPTreeIndex &GetInstance();

private:
    RPTreeIndex() {}
    ~RPTreeIndex() {}
};

#endif