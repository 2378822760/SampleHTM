//
// Created by 曹 on 2023/7/11.
//
#include <algorithm>
#include <iostream>
#include "SDRClassifier.hpp"

// #define DEBUG

SDRClassifier::SDRClassifier(const vector<UInt>& steps, UInt cellsNum,
    UInt buketNum, Real64 lr)
    : steps_(steps), cellsNum_(cellsNum), buketNum_(buketNum), lr_(lr) {
    sort(steps_.begin(), steps_.end());
    if (steps_.size() > 0) {
        maxStep_ = steps_.back();
    }
    else {
        maxStep_ = 1;
    }
    // 为每个step生成一个weight矩阵
    for (const auto& step : steps_) {
        weight_[step] = tdarray(cellsNum_, buketNum_);
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
        if (actualValIdx >= buketNum_) {
            throw runtime_error("actual value index out of range");
        }
        tdarray onehot(1, buketNum_);
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
            for (UInt j = 0; j < buketNum_; ++j) {
                weight(i, j) -= lr_ * dL_dz(0, j) *
                    input_(0, i);
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
