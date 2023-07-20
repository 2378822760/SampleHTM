/**
 * @file 实现了一个专用的softmax神经网络，采用反向传播算法才更新权重，
 * 没有偏置节点。
 */
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

    /** 构造函数
     * @brief 创建一个SDR分类器，为每一个预测步长创建创建一个单独的softmax网络。
     * @param steps 预测步长，支持多个同时预测多个步长。
     * @param cellsNum 设定在时间池(TM)中的细胞数量，在softmax网络中为输入层节点个数。
     * @param bucketNum 设定在编码器(Encoder)中桶的数量，在softmax网络中为输出节点的个数。
     * @param lr 反向传播时的学习率。
     */
    SDRClassifier(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum,
        Real64 lr);

    SDRClassifier(const map<string, string>& config);

    ~SDRClassifier() = default;

    void init(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum,
        Real64 lr);

    /** SDRClassifier的主要运行函数。
     * @brief 根据TM层获得的激活细胞，为每一个步长生成预测结果。
     * @param patternNZ 由时间池(TM)计算得出的激活细胞索引。
     * @param actualValIdx 真实值在Encoder中对应的桶的索引。
     * @param learn 为true时，更新反向传播误差。
     * @param result 由步长索引的map，每个步长对应一个数组，数组长度为Encoder中
     * 桶的数量，数组第i项表示由当前patternNZ确定Encoder中第i个桶的概率。
     * @return 返回值保存在result中。
     */
    void compute(const vector<UInt>& patternNZ, UInt actualValIdx, bool learn,
        ClassifierResult& result);

    /**
     * @brief 对每个步长打印每个权值。
     */
    void logWeight() const;

    /**
     * @brief 对每个步长打印每个结果。
     */
    void logResult(ClassifierResult& result) const;

private:
    /**
     * @brief softmax网络的前向传播，为每个步长对应的网络都进行计算。
     * 计算公式：softmax(x · weight)，·表示点乘。
     * @param patternNZ 由时间池(TM)计算得出的激活细胞索引。
     * @param result 由步长索引的map，每个步长对应一个数组，数组长度为Encoder中
     * 桶的数量，数组第i项表示由当前patternNZ确定Encoder中第i个桶的概率。
     * @return 返回值保存在result中。
     */
    void infer_(const vector<UInt>& patternNZ, ClassifierResult& result);

    /**
     * @brief softmax网络的反向传播，为每个合法步长对应的网络都进行计算。
     * 权值更新公式：w[i][j] -= dL/dw[i][j] * lr
     * @param onehot 真实值对应的结果的独热编码(只有一个bit为1,
     * 其余为0的二进制串其长度为Encoder中桶的数量)。
     */
    void learn_(const tdarray& onehot);

    map<UInt, tdarray> weight_;
    vector<UInt> steps_;
    deque<vector<UInt>> inputHistory_;
    UInt maxStep_;
    UInt cellsNum_;
    UInt bucketNum_;
    Real64 lr_;
};

#endif //SDRCLASSIFIER_LIBRARY_H
