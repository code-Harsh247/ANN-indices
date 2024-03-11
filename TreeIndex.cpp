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
    cout << "Making KD tree";
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

RPTreeIndex *RPTreeIndex::instance = nullptr;

RPTreeIndex::RPTreeIndex() : root(nullptr) {}

RPTreeIndex *RPTreeIndex::GetInstance()
{
    if (!instance)
    {
        instance = new RPTreeIndex();
    }
    return instance;
}

std::vector<double> RPTreeIndex::randomUnitDirection(size_t dimensions)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    std::vector<double> direction(dimensions, 0.0);
    double sum = 0.0;
    for (size_t i = 0; i < dimensions; ++i)
    {
        direction[i] = dis(gen);
        sum += direction[i] * direction[i];
    }
    double norm = sqrt(sum);
    for (size_t i = 0; i < dimensions; ++i)
    {
        direction[i] /= norm;
    }
    return direction;
}

int RPTreeIndex::randomX(int k)
{
    return rand() % k;
}

RPTreeIndex::Node *RPTreeIndex::buildTree(std::vector<DataVector> &points)
{
    if (points.empty())
    {
        return nullptr;
    }

    return buildTree(points.begin(), points.end());
}

RPTreeIndex::Node *RPTreeIndex::buildTree(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end)
{
    int k = begin->getVector().size();
    std::vector<double> axis = randomUnitDirection(k);

    if (end - begin <= 2)
    {
        Node *newnode = new Node;
        newnode->axis = axis;
        newnode->left = nullptr;
        newnode->right = nullptr;
        newnode->medianval = -1;
        newnode->v = std::vector<DataVector>(begin, end); // Store the points in leaf nodes
        return newnode;
    }

    DataVector x = *(begin + randomX(end - begin));

    DataVector y;
    double maxdist = 0;

    for (auto it = begin; it != end; it++)
    {
        double dist = 0;
        dist = (*it) * x;
        if (dist > maxdist)
        {
            maxdist = dist;
            y = *it;
        }
    }

    Node *newnode = new Node;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    double delta = dis(gen) * 6 * (x.dist(y)) / sqrt(k);

    newnode->axis = axis;
    std::sort(begin, end, [axis](const DataVector &a, const DataVector &b)
              {DataVector Axis;
             Axis.setValues(axis);
             return a*Axis < b*Axis; });

    auto medianiter = begin + (end - begin) / 2;
    DataVector Axis;
    Axis.setValues(axis);
    newnode->medianval = *medianiter * Axis + delta;
    for (auto it = begin; it != end; it++)
    {
        newnode->v.push_back(*it);
    }

    newnode->left = buildTree(begin, medianiter + 1);
    newnode->right = buildTree(medianiter + 1, end);

    return newnode;
}

std::vector<DataVector> RPTreeIndex::search(DataVector &point, Node *node, int k)
{
    if (node == nullptr)
    {
        return std::vector<DataVector>();
    }
    if (node->medianval == -1)
    {
        return node->v;
    }
    DataVector Node_axis;
    Node_axis.setValues(node->axis);
    double compareval = point * (Node_axis);
    std::vector<DataVector> nneighbours, sibling;
    Node *temp;

    if (compareval <= node->medianval)
    {
        temp = node->left;
        nneighbours = search(point, node->left, k);
        sibling = node->right->v;
    }
    else
    {
        temp = node->right;
        nneighbours = search(point, node->right, k);
        sibling = node->left->v;
    }

    double maxdist = -1;
    for (auto it = nneighbours.begin(); it != nneighbours.end(); it++)
    {
        if (it->dist(point) > maxdist)
        {
            maxdist = it->dist(point);
        }
    }
    DataVector temp_axis;
    temp_axis.setValues(temp->axis);
    double mediandist = abs(node->medianval - point * temp_axis);

    if (maxdist > mediandist || nneighbours.size() < k)
    {
        for (auto it = sibling.begin(); it != sibling.end(); it++)
        {
            nneighbours.push_back(*it);
        }
    }
    return nneighbours;
}
void RPTreeIndex::maketree(vector<DataVector> &points)
{
    root = buildTree(points);
}

vector<DataVector> RPTreeIndex:: query_search(DataVector &point, int k)
{
    return search(point, root, k);
}


