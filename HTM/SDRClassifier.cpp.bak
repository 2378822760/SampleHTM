#pragma once
#include <algorithm>
#include <iostream>
#include "SDRClassifier.hpp"
#include "HelpFunc.hpp"
// #define DEBUG

SDRClassifier::SDRClassifier(const vector<UInt>& steps, UInt cellsNum,
    UInt bucketNum, Real64 lr) {
    init(steps, cellsNum, bucketNum, lr);
}

SDRClassifier::SDRClassifier(const map<string, string>& config) {
    vector<UInt> steps = stov(config.at("steps"));
    UInt cellsNum = stoul(config.at("cellsNum"));
    UInt bucketNum = stoul(config.at("bucketNum"));
    Real64 lr = stod(config.at("lr"));
    init(steps, cellsNum, bucketNum, lr);
}

void SDRClassifier::init(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum,
    Real64 lr) {
    steps_     = steps;
    cellsNum_  = cellsNum;
    bucketNum_ = bucketNum;
    lr_        = lr;
#ifdef DEBUG
    cout << cellsNum_ << endl;
    cout << bucketNum_ << endl;
    cout << lr_ << endl;
    cout << steps_.size() << endl;
#endif // DEBUG

    sort(steps_.begin(), steps_.end());
    if (steps_.size() > 0) {
        maxStep_ = steps_.back();
    }
    else {
        maxStep_ = 1;
    }
    
    // 为每个step生成一个weight矩阵
    for (const auto& step : steps_) {
        weight_[step] = tdarray(cellsNum_, bucketNum_);
        weight_[step].random();
    }
}

void SDRClassifier::compute(const vector<UInt>& patternNZ, UInt actualValIdx,
    bool learn, ClassifierResult& result) {
    if (inputHistory_.size() == maxStep_) {
        inputHistory_.pop_front();
    }
    inputHistory_.emplace_back(patternNZ.begin(), patternNZ.end());
    if (learn) {
        if (actualValIdx >= bucketNum_) {
            throw runtime_error("actual value index out of range");
        }
        tdarray onehot(1, bucketNum_);
        onehot(0, actualValIdx) = 1;
        learn_(onehot);
    }
    infer_(patternNZ, result);
}

void SDRClassifier::infer_(const vector<UInt>& patternNZ, ClassifierResult& result) {
    result.clear(); // 清空result
    for (const auto& step : steps_) {
        tdarray input_(1, cellsNum_);
        for (auto i : patternNZ) {
            input_(0, i) = 1;
        }
        result[step] = tdarray();
        input_.dot(weight_[step], result[step]);
        result[step].softmax(0);
    }
#ifdef DEBUG
    logResult(result);
#endif
}

void SDRClassifier::learn_(const tdarray& onehot) {
    const auto& patternNZ = inputHistory_.back();
    tdarray input_(1, cellsNum_);
    for (auto i : patternNZ) {
        input_(0, i) = 1;
    }
    for (const auto& step : steps_) {
        if (inputHistory_.size() < step) {
            continue;
        }
        auto& weight = weight_[step];
        tdarray result;
        input_.dot(weight, result);
        result.softmax(0);
        tdarray dL_dz = result - onehot;
        // weight - x.T · dL_dz
        for (UInt i = 0; i < cellsNum_; ++i) {
            for (UInt j = 0; j < bucketNum_; ++j) {
                weight(i, j) -= lr_ * dL_dz(0, j) * input_(0, i);
            }
        }
    }
}

void SDRClassifier::logWeight() const {
    for (const auto& step : steps_) {
        cout << "step: " << step << endl;
        weight_.at(step).log();
    }
}

void SDRClassifier::logResult(ClassifierResult& result) const {
    for (const auto& step : steps_) {
        cout << "step: " << step << endl;
        result.at(step).log();
    }
}
