#include <random>
#include <algorithm>
#include <iostream>
#include <cassert>
#include "SpatialPooler.hpp"
#include "Topology.hpp"
#include "HelpFunc.hpp"
// #define DEBUG
using namespace std;
using namespace topology;

std::random_device rd;
std::mt19937_64 gen(rd()); // 使用64位版本的Mersenne Twister引擎
std::uniform_real_distribution<> dis(0.0, 1.0); // 生成0到1之间的均匀分布实数

SpatialPooler::SpatialPooler(vector<UInt> inputDimensions, vector<UInt> columnDimensions,
    UInt potentialRadius, Real potentialPct,
    bool globalInhibition, Real localAreaDensity,
    UInt numActiveColumnsPerInhArea,
    UInt stimulusThreshold, Real synPermInactiveDec,
    Real synPermActiveInc, Real synPermConnected,
    Real minPctOverlapDutyCycles,
    UInt dutyCyclePeriod, Real boostStrength) {
    init(inputDimensions, columnDimensions, potentialRadius, potentialPct,
        globalInhibition, localAreaDensity, numActiveColumnsPerInhArea,
        stimulusThreshold, synPermInactiveDec, synPermActiveInc, synPermConnected,
        minPctOverlapDutyCycles, dutyCyclePeriod, boostStrength);
}

SpatialPooler::SpatialPooler(const map<string, string>& config) {
    vector<UInt> inputDimensions = stov(config.at("inputDimensions"));
    vector<UInt> columnDimensions = stov(config.at("columnDimensions"));
    UInt potentialRadius = stoul(config.at("potentialRadius"));
    UInt dutyCyclePeriod = stoul(config.at("dutyCyclePeriod"));
    UInt numActiveColumnsPerInhArea = stoul(config.at("numActiveColumnsPerInhArea"));
    UInt stimulusThreshold = stoul(config.at("stimulusThreshold"));
    Real potentialPct = stof(config.at("potentialPct"));
    Real localAreaDensity = stof(config.at("localAreaDensity"));
    Real synPermInactiveDec = stof(config.at("synPermInactiveDec"));
    Real synPermActiveInc = stof(config.at("synPermActiveInc"));
    Real synPermConnected = stof(config.at("synPermConnected"));
    Real minPctOverlapDutyCycles = stof(config.at("minPctOverlapDutyCycles"));
    Real boostStrength = stof(config.at("boostStrength"));
    bool globalInhibition = config.at("globalInhibition") == "true";
    init(inputDimensions, columnDimensions, potentialRadius, potentialPct,
        globalInhibition, localAreaDensity, numActiveColumnsPerInhArea,
        stimulusThreshold, synPermInactiveDec, synPermActiveInc, synPermConnected,
        minPctOverlapDutyCycles, dutyCyclePeriod, boostStrength);
}

void SpatialPooler::init(vector<UInt> inputDimensions, vector<UInt> columnDimensions,
    UInt potentialRadius, Real potentialPct,
    bool globalInhibition, Real localAreaDensity,
    UInt numActiveColumnsPerInhArea,
    UInt stimulusThreshold, Real synPermInactiveDec,
    Real synPermActiveInc, Real synPermConnected,
    Real minPctOverlapDutyCycles,
    UInt dutyCyclePeriod, Real boostStrength) {
    // 1. 计算输入bit数
    inputDimensions_ = inputDimensions;
    numInputs_ = 1;
    for (auto& dim : inputDimensions) {
        numInputs_ *= dim;
    }

    // 2. 计算columns数量
    columnDimensions_ = columnDimensions;
    numColumns_ = 1;
    for (auto& dim : columnDimensions) {
        numColumns_ *= dim;
    }
    assert(numColumns_ > 0);
    assert(numInputs_ > 0);
    assert(inputDimensions_.size() == columnDimensions_.size());
    assert(numActiveColumnsPerInhArea > 0 ||
        (localAreaDensity > 0 && localAreaDensity <= 0.5));
    assert(potentialPct > 0 && potentialPct <= 1);

    potentialRadius_ = potentialRadius > numInputs_ ? numInputs_ : potentialRadius;;
    potentialPct_ = potentialPct;
    globalInhibition_ = globalInhibition;
    localAreaDensity_ = localAreaDensity;
    numActiveColumnsPerInhArea_ = numActiveColumnsPerInhArea;
    stimulusThreshold_ = stimulusThreshold;
    synPermInactiveDec_ = synPermInactiveDec;
    synPermActiveInc_ = synPermActiveInc;
    synPermConnected_ = synPermConnected;
    dutyCyclePeriod_ = dutyCyclePeriod;
    boostStrength_ = boostStrength;
    minPctOverlapDutyCycles_ = minPctOverlapDutyCycles;
    iterationNum_ = 0;
    updatePeriod_ = 50;
    initConnectedPct_ = 0.5;
    // 3. 为一些数组分配空间
    overlap_.assign(numColumns_, 0);
    boostedOverlaps_.assign(numColumns_, 0);
    boostFactors_.assign(numColumns_, 1.0);
    // neighbours_.assign(numColumns_, vector<UInt>(4));
    activeDutyCycle_.assign(numColumns_, 0);
    overlapDutyCycle_.assign(numColumns_, 0);
    minOverlapDutyCycles_.assign(numColumns_, 0);
    // activeCount_.assign(numColumns_, 0);
    // overlapCount_.assign(numColumns_, 0);

    // 4. 为每个column分配突触
    // 4.1 assign receptive field
    // 4.2 assign synapses
    for (int i = 0; i < numColumns_; ++i) {
        vector<UInt> potential = mapPotential_(i, true);
        assignSynapses_(potential);
    }
    // 4.3 compute inhibitionRadius
    updateInhibitionRadius_();

    // 4.4 get neighbour
    updateNeighbour_();
}

vector<UInt> SpatialPooler::compute(const vector<UInt>& inputVector, bool learn) {
#ifdef DEBUG
    cout << "compute" << endl;
#endif // DEBUG
    // 更新dutyCycle时要用到
    iterationNum_++;

    calculateOverlap_(inputVector);

    if (learn) boostOverlap_();
    else boostedOverlaps_.assign(overlap_.begin(), overlap_.end());
    vector<UInt> activeVector = inhibitColumns_();

    if (learn) {
        adaptSynapses_();
        updateDutyCycles_(activeVector);
        updateBoostFactors_();
        bumpUpWeakColumns_();
        if (iterationNum_ % updatePeriod_ == 0) {
            updateInhibitionRadius_();
            updateNeighbour_();
            updateMinDutyCycles_();
        }
    }
    return activeVector;
}

UInt SpatialPooler::mapColumn_(UInt column) {
    vector<UInt> columnCoords;
    CoordinateConverterND columnConv(columnDimensions_);
    columnConv.toCoord(column, columnCoords);

    vector<UInt> inputCoords;
    inputCoords.reserve(columnCoords.size());
    for (UInt i = 0; i < columnCoords.size(); i++) {
        const Real inputCoord = ((Real)columnCoords[i] + 0.5) *
            (inputDimensions_[i] / (Real)columnDimensions_[i]);

        inputCoords.push_back(floor(inputCoord));
    }

    CoordinateConverterND inputConv(inputDimensions_);
    return inputConv.toIndex(inputCoords);
}

vector<UInt> SpatialPooler::mapPotential_(UInt column, bool wrapAround) {
    const UInt centerInput = mapColumn_(column);

    vector<UInt> columnInputs;
    if (wrapAround) {
        for (UInt input : WrappingNeighborhood(centerInput, potentialRadius_,
            inputDimensions_)) {
            columnInputs.push_back(input);
        }
    }
    else {
        for (UInt input :
        Neighborhood(centerInput, potentialRadius_, inputDimensions_)) {
            columnInputs.push_back(input);
        }
    }

    UInt numPotential = round(columnInputs.size() * potentialPct_);

    vector<UInt> selectedInputs(numPotential, 0);
    sample(&columnInputs.front(), columnInputs.size(),
        &selectedInputs.front(), numPotential);


    return selectedInputs;
    /*vector<UInt> potential(numInputs_, 0);
    for (UInt input : selectedInputs) {
        potential[input] = 1;
    }

    return potential;*/
}

void SpatialPooler::mapIndexE_(const UInt beg, const UInt end, const UInt size,
    const UInt mapRadius, vector<vector<UInt>>& container) {
#ifdef DEBUG
    cout << "mapIndexE_" << endl;
#endif // DEBUG
    for (int i = beg; i < end; ++i) {
        vector<UInt> temp;
        int left = i - mapRadius, right = i + mapRadius;
        if (left < 0) {
            temp = { 0, UInt(right) + 1, size + left, size };
        }
        else if (right >= size) {
            temp = { UInt(left), size, 0, right - size + 1 };
        }
        else {
            temp = { UInt(left), UInt(right) + 1 };
        }
        container.emplace_back(std::move(temp));
    }
}

void SpatialPooler::mapIndex_(const UInt sourceSize, const UInt targetSize,
    UInt mapRadius, vector<vector<UInt>>& container) {
#ifdef DEBUG
    cout << "mapIndex_" << endl;
#endif // DEBUG
    container.clear();
    if (2 * mapRadius + 1 >= targetSize) {
        container.assign(sourceSize, { 0, targetSize });
        return;
    }
    if ((mapRadius * 2 + 1) * sourceSize < targetSize) {
        cerr << "(mapRadius * 2 + 1) * sourceSize < targetSize无法产生公平映射" << endl;
        throw logic_error("-1");
    }
    if (targetSize == sourceSize) {
        mapIndexE_(0, targetSize, targetSize, mapRadius, container);
    }
    else if (targetSize < sourceSize) {
        int coverCount = sourceSize / targetSize;
        int remainCount = sourceSize % targetSize;
        vector<vector<UInt>> temp;
        mapIndexE_(0, targetSize, targetSize, mapRadius, temp);
        for (int i = 0; i < coverCount; ++i) {
            container.insert(container.end(), temp.begin(), temp.end());
        }
        UInt beg = (targetSize - remainCount) / 2;
        mapIndexE_(beg, beg + remainCount, targetSize, mapRadius, container);
    }
    else {

    }
}


void SpatialPooler::assignSynapses_(vector<UInt>& receptiveField) {
#ifdef DEBUG
    cout << "assignSynapses_" << endl;
#endif // DEBUG
    vector<synapse> connectedSyn, potentialSyn;
    for (auto& idx : receptiveField) {
        synapse syn(idx);
        if (dis(gen) <= initConnectedPct_) {
            syn.permanence = synPermConnected_ + (1.0 - synPermConnected_) * dis(gen);
            connectedSyn.push_back(syn);
        }
        else
            syn.permanence = synPermConnected_ * dis(gen);
        potentialSyn.push_back(syn);
    }
    potentialSyn_.emplace_back(std::move(potentialSyn));
    connectedSyn_.emplace_back(std::move(connectedSyn));
}

void SpatialPooler::updateInhibitionRadius_() {
#ifdef DEBUG
    cout << "updateInhibitionRadius_" << endl;
#endif // DEBUG
    if (globalInhibition_) {
        inhibitionRadius_ = numColumns_;
        return;
    }
    Real numConnectedSyn = 0.0;
    for (auto& col : connectedSyn_) numConnectedSyn += col.size();
    numConnectedSyn /= numColumns_;
    Real columnsPerInput = (Real)numColumns_ / (Real)numInputs_;
    Real diameter = numConnectedSyn * columnsPerInput;
    Real radius = (diameter - 1) / 2.0;
    radius = max((Real)1.0, radius);
    inhibitionRadius_ = UInt(round(radius));
    cout << "numConnectedSyn : " << numConnectedSyn << endl;
}

void SpatialPooler::calculateOverlap_(const vector<UInt>& inputVector) {
#ifdef DEBUG
    cout << "calculateOverlap_" << endl;
#endif // DEBUG
    for (int i = 0; i < numColumns_; ++i) {
        int tmp = 0;
        for (auto& syn : connectedSyn_[i]) {
            tmp += inputVector[syn.sourceInputIndex];
        }
        overlap_[i] = tmp;
    }
}

void SpatialPooler::boostOverlap_() {
#ifdef DEBUG
    cout << "boostOverlap_" << endl;
#endif // DEBUG
    for (int i = 0; i < numColumns_; ++i) {
        boostedOverlaps_[i] = overlap_[i] * boostFactors_[i];
    }
}

void SpatialPooler::updateNeighbour_() {
#ifdef DEBUG
    cout << "updateNeighbour_" << endl;
#endif // DEBUG
    mapIndex_(numColumns_, numColumns_, inhibitionRadius_, neighbours_);
}

vector<UInt> SpatialPooler::inhibitColumns_() {
#ifdef DEBUG
    cout << "inhibitColumns_" << endl;
#endif // DEBUG
    int numActive = localAreaDensity_ > 0 ? (inhibitionRadius_ * 2 + 1)
        * localAreaDensity_ : numActiveColumnsPerInhArea_;
    activeColumns_.clear();
    vector<UInt> activeVector(numColumns_);
    Real globalNumActive = kthScore(0, numActive);
    bool isGlobalInhibition = (globalInhibition_ == true) || (inhibitionRadius_ >
        *max_element(columnDimensions_.begin(), columnDimensions_.end()));

    int activeVectorCnt = 0;
    for (int i = 0; i < numColumns_ && activeVectorCnt < numActive; ++i) {
        Real minLocalActivity =
            isGlobalInhibition == true ? globalNumActive : kthScore(i, numActive);
        if (boostedOverlaps_[i] > stimulusThreshold_) {
            // overlapCount_[i]++;
            if (boostedOverlaps_[i] >= minLocalActivity) {
                activeColumns_.push_back(i);
                // activeCount_[i]++;
                activeVector[i] = 1;
                activeVectorCnt++;
            }
        }
    }
    return activeVector;
}

Real SpatialPooler::kthScore(UInt colIndex, UInt K) {
#ifdef DEBUG
    cout << "kthScore" << endl;
#endif // DEBUG
    vector<Real> neighboursOverlap;
    int numRange = neighbours_[colIndex].size() / 2;
    for (int i = 0; i < numRange; ++i) {
        for (int j = neighbours_[colIndex][i * 2]; j < neighbours_[colIndex][i * 2 + 1]; ++j) {
            neighboursOverlap.push_back(boostedOverlaps_[j]);
        }
    }
    sort(neighboursOverlap.begin(), neighboursOverlap.end());
    return neighboursOverlap[K - 1];
}

void SpatialPooler::adaptSynapses_() {
#ifdef DEBUG
    cout << "adaptSynapses_" << endl;
#endif // DEBUG
    for (auto columnIndex : activeColumns_) {
        for (auto& syn : potentialSyn_[columnIndex]) {
            if (syn.permanence > synPermConnected_) {
                syn.permanence += synPermActiveInc_;
                syn.permanence = min(Real(1), syn.permanence);
            }
            else {
                syn.permanence -= synPermInactiveDec_;
                syn.permanence = max(Real(0), syn.permanence);
            }
        }
    }
}

void SpatialPooler::updateDutyCycles_(const vector<UInt>& activeVector) {
#ifdef DEBUG
    cout << "updateDutyCycles_" << endl;
#endif // DEBUG
    UInt period =
        dutyCyclePeriod_ > iterationNum_ ? iterationNum_ : dutyCyclePeriod_;
    for (int i = 0; i < numColumns_; ++i) {
        activeDutyCycle_[i] =
            (activeDutyCycle_[i] * (period - 1) + (overlap_[i] > 0 ? 1 : 0)) / period;
        overlapDutyCycle_[i] =
            (overlapDutyCycle_[i] * (period - 1) + (activeVector[i] > 0 ? 1 : 0)) / period;
    }
}

void SpatialPooler::updateBoostFactors_() {
#ifdef DEBUG
    cout << "updateBoostFactors_" << endl;
#endif // DEBUG
    Real activeDutyCycleNeighbors = 0;
    for (int idx = 0; idx < numColumns_; ++idx) {
        int numRange = neighbours_[idx].size() / 2;
        Real sumDutyCycle = 0;
        for (int i = 0; i < numRange; ++i) {
            for (int j = neighbours_[idx][i * 2]; j < neighbours_[idx][i * 2 + 1]; ++j) {
                sumDutyCycle += activeDutyCycle_[j];
            }
        }
        activeDutyCycleNeighbors = sumDutyCycle / (2 * inhibitionRadius_ + 1);
        boostFactors_[idx] =
            exp(boostStrength_ * (activeDutyCycleNeighbors - activeDutyCycle_[idx]));
    }

}

void SpatialPooler::bumpUpWeakColumns_() {
#ifdef DEBUG
    cout << "bumpUpWeakColumns_" << endl;
#endif // DEBUG
    for (int i = 0; i < numColumns_; ++i) {
        if (overlapDutyCycle_[i] < minOverlapDutyCycles_[i]) {
            increasePermanences_(i, 0.1);
        }
    }
}

void SpatialPooler::updateMinDutyCycles_() {
#ifdef DEBUG
    cout << "updateMinDutyCycles_" << endl;
#endif // DEBUG
    if (globalInhibition_) {
        Real maxOverlapDutyCycles =
            *max_element(overlapDutyCycle_.begin(), overlapDutyCycle_.end());
        fill(minOverlapDutyCycles_.begin(), minOverlapDutyCycles_.end(),
            minPctOverlapDutyCycles_ * maxOverlapDutyCycles);
        return;
    }
    // 非全局抑制更新MinDutyCycles，效率很低
    for (int i = 0; i < numColumns_; ++i) {
        vector<Real> neighbourOverlapDutyCycles;
        neighbourOverlapDutyCycles.reserve(2 * inhibitionRadius_ + 1);
        int numRange = neighbours_[i].size() / 2;
        for (int i = 0; i < numRange; ++i) {
            for (int j = neighbours_[i][i * 2]; j < neighbours_[i][i * 2 + 1]; ++j) {
                neighbourOverlapDutyCycles.push_back(overlapDutyCycle_[j]);
            }
        }
        Real maxOverlapDutyCycles =
            *max_element(neighbourOverlapDutyCycles.begin(), neighbourOverlapDutyCycles.end());
        minOverlapDutyCycles_[i] = minPctOverlapDutyCycles_ * maxOverlapDutyCycles;
    }
}

void SpatialPooler::increasePermanences_(UInt columnIndex, Real incFactor) {
#ifdef DEBUG
    cout << "increasePermanences_" << endl;
#endif // DEBUG
    for (auto& syn : potentialSyn_[columnIndex]) {
        syn.permanence += synPermConnected_ * incFactor;
    }
}

void SpatialPooler::printParameters() const {
    std::cout << "------------CPP SpatialPooler Parameters ------------------\n";
    std::cout
        << "iterationNum                = " << iterationNum_ << std::endl
        << "numInputs                   = " << numInputs_ << std::endl
        << "numColumns                  = " << numColumns_ << std::endl
        << "localAreaDensity            = " << localAreaDensity_ << std::endl
        << "numActiveColumnsPerInhArea  = " << numActiveColumnsPerInhArea_
        << std::endl
        << "potentialRadius             = " << potentialRadius_ << endl
        << "potentialPct                = " << potentialPct_ << std::endl
        << "initConnectedPct            = " << initConnectedPct_ << endl
        << "globalInhibition            = " << globalInhibition_ << std::endl
        << "inhibitionRadius            = " << inhibitionRadius_ << std::endl
        << "stimulusThreshold           = " << stimulusThreshold_ << std::endl
        << "synPermActiveInc            = " << synPermActiveInc_ << std::endl
        << "synPermInactiveDec          = " << synPermInactiveDec_ << std::endl
        << "synPermConnected            = " << synPermConnected_ << std::endl
        << "minPctOverlapDutyCycles     = " << minPctOverlapDutyCycles_
        << std::endl
        << "dutyCyclePeriod             = " << dutyCyclePeriod_ << std::endl
        << "boostStrength               = " << boostStrength_ << std::endl;
}