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
TreeIndex &TreeIndex::GetInstance()
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
KDTreeIndex &KDTreeIndex::GetInstance()
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

pair<VectorDataset, VectorDataset> KDTreeIndex::ChooseRule()
{
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
int KDTreeIndex::maxSpreadAxis()
{
    int dimensions = Data.getDimension();
    double maxVariance = -1;
    int maxAxis = -1;

    for (int axis = 0; axis < dimensions; ++axis)
    {
        // Calculate mean along current axis
        double mean = 0;
        for (const auto &point : Data.getDataset())
        {
            mean += point.getVector()[axis];
        }
        mean /= Data.getDataset().size();

        // Calculate variance along current axis
        double variance = 0;
        for (const auto &point : Data.getDataset())
        {
            double diff = point.getVector()[axis] - mean;
            variance += diff * diff;
        }
        variance /= (Data.getDataset().size());

        if (variance > maxVariance)
        {
            maxVariance = variance;
            maxAxis = axis;
        }
    }

    return maxAxis;
}

KDTreeIndex *KDTreeIndex::MakeTree()
{
    if (Data.getDataset().size() <= 100)
    {
        return this;
    }
    // Choose splitting axis based on maximum spread
    std::pair<VectorDataset, VectorDataset> leftRight = ChooseRule();
    KDTreeIndex *leftChild = new KDTreeIndex();
    KDTreeIndex *rightChild = new KDTreeIndex();

    // Add data to left child
    for (const auto &data : leftRight.first.getDataset())
    {
        leftChild->Data = leftRight.first;
    }

    // Add data to right child
    for (const auto &data : leftRight.second.getDataset())
    {
        rightChild->Data = leftRight.second;
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

int KDTreeIndex::getAxis() const
{
    return axis;
}

double KDTreeIndex::getMedian() const
{
    return median;
}

double KDTreeIndex::distanceToHyperplane(const DataVector &query, const KDTreeIndex *node)
{
    int axis = node->getAxis();
    double median = node->getMedian();
    return abs(query.getVector()[axis] - median);
}

vector<DataVector> KDTreeIndex::Search(const DataVector &query, int k)
{
    // If leaf node is reached, compute nearest neighbors using kNearestNeighbor function
    if (left == nullptr && right == nullptr)
    {
        VectorDataset nearestNeighbors = Data.kNearestNeighbor(query, k);
        return nearestNeighbors.getDataset();
    }

    // Search left or right subtree based on query point's position
    vector<DataVector> result;
    if (query.getVector()[axis] <= this->median)
    {
        result = left->Search(query, k);
    }
    else
    {
        result = right->Search(query, k);
    }

    // Backtrack and explore sibling node if necessary
    double distanceToBoundary = distanceToHyperplane(query, this);
    for (const auto &data : result)
    {
        double distToData = data.dist(query);
        // Check if current farthest point is nearer than the distance from the boundary with the sibling
        if (result.size() < k || distToData < distanceToBoundary)
        {
            // Explore sibling node's region
            vector<DataVector> siblingResult;
            if (query.getVector()[axis] <= this->median)
            {
                siblingResult = right->Search(query, k);
            }
            else
            {
                siblingResult = left->Search(query, k);
            }
            // Update nearest neighbors list if necessary
            for (const auto &siblingData : siblingResult)
            {
                double distToSiblingData = siblingData.dist(query);
                if (result.size() < k || distToSiblingData < distToData)
                {
                    result.push_back(siblingData);
                    // Keep the list sorted by distance from the query point
                    std::sort(result.begin(), result.end(), [&](const DataVector &a, const DataVector &b)
                              { return a.dist(query) < b.dist(query); });
                    // Ensure the list has at most k elements
                    if (result.size() > k)
                    {
                        result.pop_back();
                    }
                    // Update distance to boundary for further pruning
                    distanceToBoundary = result.back().dist(query);
                }
            }
        }
        else
        {
            // No need to explore sibling region, backtrack
            break;
        }
    }

    return result;
}

// Create Instance
RPTreeIndex &RPTreeIndex::GetInstance()
{
    static RPTreeIndex instance;
    return instance;
}

RPTreeIndex::~RPTreeIndex()
{
    if (left != NULL)
        delete left;
    if (right != NULL)
        delete right;
}

DataVector RPTreeIndex::RandomUnitDirection()
{
    // Generate random values for each dimension of the vector
    vector<double> values;
    double randomDouble = -1 + static_cast<double>(rand()) / (RAND_MAX / 2);
    for (int i = 0; i < Data.getDataset()[0].getDimension(); ++i)
    {
        values.push_back(randomDouble);
    }

    // Normalize the vector to obtain a unit direction vector
    double length = sqrt(inner_product(values.begin(), values.end(), values.begin(), 0.0));
    vector<double> normalized_values;
    for (double val : values)
    {
        normalized_values.push_back(val / length);
    }
    DataVector result;
    result.setValues(normalized_values);
    return result;
}

// Function to find the farthest point from a given point in the dataset
DataVector RPTreeIndex::FindFarthestPoint(const DataVector &point)
{
    double maxDistance = 0;
    DataVector farthestPoint;

    // Iterate through all points in the dataset
    for (int i = 0; i < Data.getDataset().size(); ++i)
    {
        DataVector currentPoint = Data.getDataset()[i];
        // Calculate the distance between the current point and the given point
        double distance = point.dist(currentPoint);
        // Update the farthest point if needed
        if (distance > maxDistance)
        {
            maxDistance = distance;
            farthestPoint = currentPoint;
        }
    }
    return farthestPoint;
}

DataVector RPTreeIndex::SelectRandomPoint()
{
    // Randomly select an index within the range of the dataset
    int index = rand() % Data.getDataset().size();
    // Get the data vector at the selected index
    return Data.getDataset()[index];
}

double RPTreeIndex::CalculateProjection(const DataVector &point, const DataVector &direction) const
{
    // Calculate the dot product of the point and the direction vector
    return inner_product(point.getVector().begin(), point.getVector().end(), direction.getVector().begin(), 0.0);
}

double RPTreeIndex::CalculateMedianProjection(const DataVector &direction) const
{
    vector<double> projections;

    // Calculate the projection of each point in S onto the direction vector
    for (const auto &data : Data.getDataset())
    {
        double projection = CalculateProjection(data, direction);
        projections.push_back(projection);
    }

    // Sort the projections vector
    sort(projections.begin(), projections.end());

    // Calculate the median value of the sorted projections
    return projections[projections.size() / 2];
}

double RPTreeIndex::getThreshold() const
{
    return threshold;
}

pair<VectorDataset, VectorDataset> RPTreeIndex::ChooseRule()
{
    // Step 1: Choose a random unit direction v
    DataVector v = RandomUnitDirection();
    randomDirection.setValues(v.getVector());

    // Step 2: Select a random point x from the dataset S
    DataVector x = SelectRandomPoint();

    // Step 3: Find the farthest point y from x in S
    DataVector y = FindFarthestPoint(x);

    // Step 4: Choose a random offset δ
    double delta = (double)rand() / RAND_MAX * 2 - 1;

    // Step 5: Define the partitioning rule
    auto Rule = [&](const DataVector &point)
    {
        // Calculate the projection of the point onto the direction v
        double projection = CalculateProjection(point, v);

        // Calculate the threshold value based on the median of projections and the random offset δ
        median = CalculateMedianProjection(v);
        threshold = median + delta;

        // Compare the projection with the threshold value to determine the partitioning
        return projection <= threshold;
    };

    // Step 6: Apply the partitioning rule to split the dataset into two subsets
    VectorDataset leftSubset, rightSubset;
    for (const auto &data : Data.getDataset())
    {
        if (Rule(data))
        {
            leftSubset.pushValue(data);
        }
        else
        {
            rightSubset.pushValue(data);
        }
    }

    // Step 7: Return a pair of VectorDatasets containing the split dataset
    return {leftSubset, rightSubset};
}

RPTreeIndex *RPTreeIndex::MakeTree()
{
    // Base case: if dataset size is below threshold, return leaf node
    if (Data.getDataset().size() <= 100)
    {
        return this;
    }

    pair<VectorDataset, VectorDataset> leftRight = ChooseRule();
    RPTreeIndex *leftChild = new RPTreeIndex();
    RPTreeIndex *rightChild = new RPTreeIndex();

    for (const auto &data : leftRight.first.getDataset())
    {
        leftChild->AddData(data);
    }

    // Add data to right child
    for (const auto &data : leftRight.second.getDataset())
    {
        rightChild->AddData(data);
    }

    // Recursively build left and right subtrees
    left = leftChild->MakeTree();
    right = rightChild->MakeTree();

    return this;
}

DataVector RPTreeIndex::getDirection() const
{
    return randomDirection;
}

double RPTreeIndex::calculateDistanceToHyperplane(RPTreeIndex* node, const DataVector &query) const {
    // Calculate the projection of the query point onto the hyperplane at the current node
    double projection = CalculateProjection(query, node->randomDirection);
    
    // Calculate the distance from the query point to the hyperplane
    return abs(projection - node->median);
}


vector<DataVector> RPTreeIndex::Search(const DataVector &query, int k)
{
    // If leaf node is reached, compute nearest neighbors using kNearestNeighbor function
    if (left == nullptr && right == nullptr)
    {
        VectorDataset nearestNeighbors = Data.kNearestNeighbor(query, k);
        return nearestNeighbors.getDataset();
    }

    // Search left or right subtree based on query point's position
    vector<DataVector> result;
    double projection = CalculateProjection(query, getDirection());
    if (projection <= threshold)
    {
        result = left->Search(query, k);
    }
    else
    {
        result = right->Search(query, k);
    }

    // Backtrack and explore sibling node if necessary
    double distanceToBoundary = calculateDistanceToHyperplane(this, query);
    for (const auto &data : result)
    {
        double distToData = query.dist(data);
        // Check if current farthest point is nearer than the distance from the boundary with the sibling
        if (result.size() < k || distToData < distanceToBoundary)
        {
            // Explore sibling node's region
            vector<DataVector> siblingResult;
            if (projection <= threshold)
            {
                siblingResult = right->Search(query, k);
            }
            else
            {
                siblingResult = left->Search(query, k);
            }
            // Update nearest neighbors list if necessary
            for (const auto &siblingData : siblingResult)
            {
                double distToSiblingData = siblingData.dist(query);
                if (result.size() < k || distToSiblingData < distToData)
                {
                    result.push_back(siblingData);
                    // Keep the list sorted by distance from the query point
                    std::sort(result.begin(), result.end(), [&](const DataVector &a, const DataVector &b)
                              { return a.dist(query) < b.dist(query); });
                    // Ensure the list has at most k elements
                    if (result.size() > k)
                    {
                        result.pop_back();
                    }
                    // Update distance to boundary for further pruning
                    distanceToBoundary = result.back().dist(query);
                }
            }
        }
        else
        {
            // No need to explore sibling region, backtrack
            break;
        }
    }

    return result;
}