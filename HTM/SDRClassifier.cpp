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
    steps_ = steps;
    cellsNum_ = cellsNum;
    bucketNum_ = bucketNum;
    lr_ = lr;
#ifdef DEBUG
    cout << cellsNum_ << endl;
    cout << bucketNum_ << endl;
    cout << lr_ << endl;
    cout << steps_.size() << endl;
#endif // DEBUG
    sort(steps_.begin(), steps_.end());
    if (steps_.size() > 0) maxStep_ = steps_.back();
    else maxStep_ = 1;
    for (auto step : steps_) 
        networks_[step] = Classifier(step, cellsNum_, bucketNum_, lr_);
}

ClassifierResult SDRClassifier::compute(const vector<UInt>& patternNZ, 
    UInt actualValIdx, bool learn) {
    // 1. �����Softmax���������x
    Eigen::MatrixXd seq = Eigen::MatrixXd::Zero(1, cellsNum_);
    for (auto activeCellIndex : patternNZ) seq(0, activeCellIndex) = 1.0;
    // 2. ά��������ʷ���У���֤������󳤶�ΪmaxStep + 1
    while (inputHistory_.size() > maxStep_) inputHistory_.pop_front();
    inputHistory_.push_back(std::move(seq));
    if (learn) {
        if (actualValIdx >= bucketNum_) {
            throw runtime_error("actual value index out of range");
        }
        // 3. ��������������Ӧ��Softmax����ʵ���
        Eigen::MatrixXd y = Eigen::MatrixXd::Zero(1, this->bucketNum_);
        y(0, actualValIdx) = 1.0;
        // 4. �������в�����Ӧ�ķ�������Ȩ��
        learn_(y, this->steps_, this->networks_, this->inputHistory_);
    }
    // 5. ǰ�򴫲�
    return infer_(inputHistory_, this->steps_, this->networks_, learn);
}

ClassifierResult SDRClassifier::infer_(const deque<Eigen::MatrixXd>& inputHistory, 
    const vector<UInt>& steps, map<UInt, Classifier>& networks, bool learn) {
    int historiesNum = inputHistory.size();
    ClassifierResult res;
    for (auto step : steps) {
        res[step] = networks[step].predict(inputHistory.back());
    }
    return res;
}

void SDRClassifier::learn_(const Eigen::MatrixXd& y, const vector<UInt>& steps,
    map<UInt, Classifier>& networks, const deque<Eigen::MatrixXd>& inputHistory) {
    int historiesNum = inputHistory.size();
    for (auto step : steps) {
        // ����������������ʷ���г��ȣ���ʱ�����ܽ��в������¡�
        if (step >= historiesNum) break;
        auto& nn = networks[step];
        // ������Ϊ1ʱ�������ٽ���һ��ǰ�򴫲����㣬ֱ�ӽ��з��򴫲�����
        if (step == 1) nn.backPropagation(y);
        else nn.update(inputHistory[historiesNum - step - 1], y);
    }
}