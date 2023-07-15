//
// Created by ²Ü on 2023/7/11.
//
#pragma once
#ifndef SDRCLASSIFIER_LIBRARY_H
#define SDRCLASSIFIER_LIBRARY_H
#include <vector>
#include <map>
#include <deque>
#include "type.hpp"
#include "tdarray.hpp"

using namespace std;

typedef map<UInt, tdarray> ClassifierResult;

class SDRClassifier {
public:
    SDRClassifier() = default;

    SDRClassifier(const vector<UInt>& steps, UInt cellsNum, UInt buketNum,
        Real64 lr);

    ~SDRClassifier() = default;

    void compute(const vector<UInt>& patternNZ, UInt actualValIdx, bool learn,
        ClassifierResult& result);

    void logWeight() const;

    void logResult(ClassifierResult& result) const;

private:
    void infer_(const vector<UInt>& patternNZ, ClassifierResult& result);

    void learn_(const tdarray& onehot);

    map<UInt, tdarray> weight_;
    vector<UInt> steps_;
    deque<vector<UInt>> inputHistory_;
    UInt maxStep_;
    UInt cellsNum_;
    UInt buketNum_;
    Real64 lr_;
};

#endif //SDRCLASSIFIER_LIBRARY_H
