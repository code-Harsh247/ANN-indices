// TreeIndex.cpp

#include "TreeIndex.h"
#include <bits/stdc++.h>

// TreeIndex Constructor function
TreeIndex::TreeIndex()
{
}

// TreeIndex Destructor function
TreeIndex::~TreeIndex()
{
}
TreeIndex& TreeIndex::GetInstance()
{
    static TreeIndex instance;
    return instance;
}

// AddData function
void TreeIndex::AddData(const DataVector &data)
{
    Data.pushValue(data);
}

// KDTreeIndex GetInstace
KDTreeIndex& KDTreeIndex::GetInstance()
{
    static KDTreeIndex instance;
    return instance;
}

// KDTreeIndex constructor function
KDTreeIndex::KDTreeIndex()
{
    left = NULL;
    right = NULL;
    median = 0;
    axis = 0;
}

// KDTreeIndex destructor function
KDTreeIndex::~KDTreeIndex()
{
    if (left != NULL)
        delete left;
    if (right != NULL)
        delete right;
}

pair<VectorDataset,VectorDataset> KDTreeIndex:: ChooseRule(){
    vector<pair<double, int>> temp;

    for (int i = 0; i < Data.getDataset().size(); i++)
    {
        temp.push_back({Data.getDataset()[i].getVector()[axis], i});
    }
    // sort temp vector in the assending order of 1st element
    sort(temp.begin(), temp.end());
    // median is the middle element of the sorted vector
    this->median = temp[temp.size() / 2].first;
    // left and right vector dataset
    VectorDataset left, right;
    for (int i = 0; i < temp.size(); i++)
    {
        if (temp[i].first <= this->median)
        {
            left.pushValue(Data.getDataVector(temp[i].second));
        }
        else
        {
            right.pushValue(Data.getDataVector(temp[i].second));
        }
    }
    return {left, right};
}
int KDTreeIndex:: maxSpreadAxis(){
    int dimensions = Data.getDimension();
    double maxVariance = -1;
    int maxAxis = -1;

    for (int axis = 0; axis < dimensions; ++axis) {
        // Calculate mean along current axis
        double mean = 0;
        for (const auto& point : Data.getDataset()) {
            mean += point.getVector()[axis];
        }
        mean /= Data.getDataset().size();

        // Calculate variance along current axis
        double variance = 0;
        for (const auto& point : Data.getDataset()) {
            double diff = point.getVector()[axis] - mean;
            variance += diff * diff;
        }
        variance /= (Data.getDataset().size());

        if (variance > maxVariance) {
            maxVariance = variance;
            maxAxis = axis;
        }
    }

    return maxAxis;
}

KDTreeIndex *KDTreeIndex:: MakeTree(){
    if (Data.getDataset().size() <= 100) {
        return this;
    }
    // Choose splitting axis based on maximum spread
    std::pair<VectorDataset, VectorDataset> leftRight = ChooseRule();
    KDTreeIndex* leftChild = new KDTreeIndex();
    KDTreeIndex* rightChild = new KDTreeIndex();

     // Add data to left child
    for (const auto& data : leftRight.first.getDataset()) {
        leftChild->AddData(data);
    }

    // Add data to right child
    for (const auto& data : leftRight.second.getDataset()) {
        rightChild->AddData(data);
    }

    // Determine next splitting axis (alternating)
    int nextAxis = maxSpreadAxis();

    // Set splitting axis for child nodes
    leftChild->axis = nextAxis;
    rightChild->axis = nextAxis;

    // Recursively build left and right subtrees
    left = leftChild->MakeTree();
    right = rightChild->MakeTree();

    return this;
}

int KDTreeIndex:: getAxis() const{
    return axis;
}

double KDTreeIndex:: getMedian() const {
    return median;
}

double KDTreeIndex:: distanceToHyperplane(const DataVector& query, const KDTreeIndex* node) {
    int axis = node->getAxis();
    double median = node->getMedian();
    return abs(query.getVector()[axis] - median);
}

vector<DataVector> KDTreeIndex::Search(const DataVector &query, int k)
{
    // If leaf node is reached, compute nearest neighbors using kNearestNeighbor function
    if (left == nullptr && right == nullptr) {
        VectorDataset nearestNeighbors = Data.kNearestNeighbor(query, k);
        return nearestNeighbors.getDataset();
    }

    // Search left or right subtree based on query point's position
    vector<DataVector> result;
    if (query.getVector()[axis] <= this->median) {
        result = left->Search(query, k);
    } else {
        result = right->Search(query, k);
    }

    // Backtrack and explore sibling node if necessary
    double distanceToBoundary = distanceToHyperplane(query, this);
    for (const auto& data : result) {
        double distToData = data.dist(query);
        // Check if current farthest point is nearer than the distance from the boundary with the sibling
        if (result.size() < k || distToData < distanceToBoundary) {
            // Explore sibling node's region
            vector<DataVector> siblingResult;
            if (query.getVector()[axis] <= this->median) {
                siblingResult = right->Search(query, k);
            } else {
                siblingResult = left->Search(query, k);
            }
            // Update nearest neighbors list if necessary
            for (const auto& siblingData : siblingResult) {
                double distToSiblingData = siblingData.dist(query);
                if (result.size() < k || distToSiblingData < distToData) {
                    result.push_back(siblingData);
                    // Keep the list sorted by distance from the query point
                    std::sort(result.begin(), result.end(), [&](const DataVector& a, const DataVector& b) {
                        return a.dist(query) < b.dist(query);
                    });
                    // Ensure the list has at most k elements
                    if (result.size() > k) {
                        result.pop_back();
                    }
                    // Update distance to boundary for further pruning
                    distanceToBoundary = result.back().dist(query);
                }
            }
        } else {
            // No need to explore sibling region, backtrack
            break;
        }
    }

    return result;
}



