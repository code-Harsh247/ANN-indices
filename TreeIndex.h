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
    ~RPTreeIndex();
    static RPTreeIndex &GetInstance();
    RPTreeIndex *MakeTree();
    pair<VectorDataset,VectorDataset> ChooseRule();
    vector<DataVector> Search(const DataVector &query, int k);
    DataVector RandomUnitDirection();
    DataVector SelectRandomPoint();
    DataVector FindFarthestPoint(const DataVector& x);
    double RandomOffset();
    double CalculateProjection(const DataVector& point, const DataVector& direction) const;
    double CalculateMedianProjection(const DataVector& direction) const;
    void setRandomDirection(const DataVector& dir);
    double calculateDistanceToHyperplane(RPTreeIndex* node, const DataVector &query) const;
    DataVector getDirection() const;
    double getThreshold() const;


private:
    DataVector randomDirection;
    double threshold;
    double median;
    VectorDataset Data;
    RPTreeIndex(){};
    RPTreeIndex *left;
    RPTreeIndex *right;
};

#endif