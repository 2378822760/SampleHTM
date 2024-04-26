#include "Encoder.hpp"
#include <iostream>
#include <algorithm>
#include <string>
using namespace std;

ScalarEncoder::ScalarEncoder(int w, double minValue, double maxValue, int bucketNum,
    bool clipInput) {
    init(w, minValue, maxValue, bucketNum, clipInput);
}

ScalarEncoder::ScalarEncoder(const map<string, string>& config) {
    // 检查配置项是否完整
    if (config.find("w") == config.end() || config.find("minValue") == config.end() ||
        config.find("maxValue") == config.end() || config.find("bucketNum") == config.end() ||
        config.find("clipInput") == config.end()) {
        cerr << "config is not complete" << endl;
        throw invalid_argument("-1");
    }
    int w = stoi(config.at("w"));
    double minValue = stod(config.at("minValue"));
    double maxValue = stod(config.at("maxValue"));
    int bucketNum = stoi(config.at("bucketNum"));
    bool clipInput = config.at("clipInput") == "true";
    init(w, minValue, maxValue, bucketNum, clipInput);
}

void ScalarEncoder::init(int w, double minValue, double maxValue, int bucketNum,
    bool clipInput) {
    if (w <= 0) {
        cerr << "w must be > 0" << endl;
        throw invalid_argument("-1");
    }

    if (bucketNum <= 0) {
        cerr << "bucketNum = " << bucketNum << "must > 0" << endl;
        throw invalid_argument("-1");
    }

    if (minValue >= maxValue) {
        cerr << "minValue must be < maxValue. minValue=" << minValue
            << " maxValue=" << maxValue << endl;
        throw invalid_argument("-1");
    }

    w_ = w;
    minValue_ = minValue;
    maxValue_ = maxValue;
    bucketNum_ = bucketNum;
    clipInput_ = clipInput;
    n_ = bucketNum + w - 1; // 计算输出宽度
    lastBucketIdx_ = 0;
}

vector<UInt> ScalarEncoder::encodeIntoArray(double input) {
    if (input < minValue_ || input > maxValue_) {
        if (clipInput_) {
            input = input < minValue_ ? minValue_ : maxValue_;
        }
        else {
            cerr << "input (" << input << ") out of range [" << minValue_
                << ", " << maxValue_ << "]" << endl;
            throw out_of_range("-1");
        }
    }
    vector<UInt> output(getOutputWidth());

    int iBucket = floor(bucketNum_ * (input - minValue_) /
        (maxValue_ - minValue_));
    if ((input - minValue_) == (maxValue_ - minValue_))
        iBucket--;
    lastBucketIdx_ = iBucket;
    // cout << iBucket << endl;
    int endedIdx = min(w_ + iBucket, (int)n_);

    fill(output.begin() + iBucket, output.begin() + endedIdx, 1);

    return output;
}

double ScalarEncoder::decode(UInt bucketIdx)
{
    if (bucketIdx >= bucketNum_) {
        cerr << "Bucket index = " << bucketIdx << " is out of range[0, "
            << bucketNum_ - 1 << "]." << endl;
        throw out_of_range("-1");
    }
    double res = bucketIdx * (maxValue_ - minValue_) / bucketNum_ + minValue_;

    return res;
}

void ScalarEncoder::printParameters() const {
    std::cout << "------------CPP ScalarEncoder Parameters ------------------\n";
    std::cout
        << "w                           = " << w_ << std::endl
        << "minValue_                   = " << minValue_ << std::endl
        << "maxValue_                   = " << maxValue_ << std::endl
        << "bucketNum_                  = " << bucketNum_ << std::endl
        << "n_                          = " << n_ << std::endl
        << "clipInput_                  = " << clipInput_ << std::endl;
}