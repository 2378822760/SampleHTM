/**
 * @file 实现了一个专用的softmax神经网络，采用反向传播算法才更新权重，
 * 没有偏置节点。
 */
#ifndef SDRCLASSIFIER_LIBRARY_H
#define SDRCLASSIFIER_LIBRARY_H
#include <vector>
#include <map>
#include <deque>
#include <string>

#include "Astral.hpp"
#include "Type.hpp"

using namespace std;

/**
 * @brief 将步长作为key存储所有结果
*/
typedef map<UInt, Eigen::MatrixXd> ClassifierResult;

/**
 * @brief 一个Softmax神经网络网络分类器
*/
class Classifier {
public:
    Classifier() = default;

    Classifier(UInt step, UInt cellsNum, UInt bucketNum, Real64 lr) :
    step_(step), cellsNum_(cellsNum), bucketNum_(bucketNum), lr_(lr) {
        nn_.add(cellsNum, nullptr, Astral::Input);
        nn_.add(bucketNum, new Astral::Softmax(), Astral::Output);
        nn_.compile(new Astral::SGD(lr), new Astral::CrossEntropyLoss());
    }

    Classifier(const Classifier&) = default;

    Classifier& operator=(const Classifier&) = default;

    Classifier& operator=(Classifier&&) = default;

    Eigen::MatrixXd predict(const Eigen::MatrixXd& x) {
        return nn_.predict(x);
    }

    void update(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
        nn_.fit(x, y, 1, 1);
    }

    void backPropagation(const Eigen::MatrixXd& target) {
        nn_.backPropagation(target);
    }

    Eigen::MatrixXd getOutput() const {
        return nn_.layers_.back().y_;
    }
private:
    UInt step_, cellsNum_, bucketNum_;

    Real64 lr_;

    Astral::Sequential nn_;

    // deque<Eigen::MatrixXd> outputHistory_;
};

class SDRClassifier {
public:
    /** 构造函数
     * @brief 创建一个SDR分类器，为每一个预测步长创建创建一个单独的softmax网络。
     * @param steps 预测步长，支持多个同时预测多个步长。
     * @param cellsNum 设定在时间池(TM)中的细胞数量，在softmax网络中为输入层节点个数。
     * @param bucketNum 设定在编码器(Encoder)中桶的数量，在softmax网络中为输出节点的个数。
     * @param lr 反向传播时的学习率。
     */
    SDRClassifier(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum, Real64 lr);

    /**
     * @brief 根据配置文件初始化
     * @param config 从配置文件读取的SDRClassifier对应的配置项。
    */
    SDRClassifier(const map<string, string>& config);

    ~SDRClassifier() = default;

    void init(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum, Real64 lr);

    /** SDRClassifier的主要运行函数。
     * @brief 根据TM层获得的激活细胞，为每一个步长生成预测结果。
     * @param patternNZ 由时间池(TM)计算得出的激活细胞索引。
     * @param actualValIdx 真实值在Encoder中对应的桶的索引。
     * @param learn 为true时，更新反向传播误差。
     * @return 由步长索引的map，每个步长对应一个数组，数组长度为Encoder中
     * 桶的数量，数组第i项表示由当前patternNZ确定Encoder中第i个桶的概率。
     */
    ClassifierResult compute(const vector<UInt>& patternNZ, UInt actualValIdx, bool learn);
private:
    /**
     * @brief softmax网络的前向传播，为每个步长对应的网络都进行计算。
     * 计算公式：softmax(x · weight)，·表示点乘。
     * @param inputHistory 输入到SDRClassifier中的历史内容，最多保存maxStep个历史输入。
     * @param steps 需要进行推断的步长列表。
     * @param networks 步长列表所对应的Softmax网络列表。
     * @return 由步长索引的map，每个步长对应一个数组，数组长度为Encoder中
     * 桶的数量，数组第i项表示由当前patternNZ确定Encoder中第i个桶的概率。
     */
    ClassifierResult infer_(const deque<Eigen::MatrixXd>& inputHistory, const vector<UInt>& steps,
        map<UInt, Classifier>& networks, bool learn);

    /**
     * @brief softmax网络的反向传播，为每个合法步长对应的网络都进行计算。
     * @param y 本次输入对应的真实值，大小为bucketNum
     * @param steps 需要进行推断的步长列表。
     * @param networks 步长列表所对应的Softmax网络列表。
     * @param inputHistory 输入到SDRClassifier中的历史内容，最多保存maxStep个历史输入。
     */
    void learn_(const Eigen::MatrixXd& y, const vector<UInt>& steps,
        map<UInt, Classifier>& networks, const deque<Eigen::MatrixXd>& inputHistory);

    vector<UInt> steps_;

    deque<Eigen::MatrixXd> inputHistory_;

    map<UInt, Classifier> networks_;

    UInt maxStep_;

    UInt cellsNum_;

    UInt bucketNum_;

    Real64 lr_;
};

#endif //SDRCLASSIFIER_LIBRARY_H
