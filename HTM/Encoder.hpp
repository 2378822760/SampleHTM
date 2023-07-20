/** @file 实现了标量编码器。
 * 编码器规则
 * 在编码数据时，有一些重要的方面需要考虑：
 * 1. 语义相似的数据应该产生活跃比特彼此重叠的 SDR
 * 2. 相同的输入数据应该总是产生相同的输出结果
 * 3. 对所有的输入信息，输出信息应该有相同的维度（比特总数）
 * 4. 对所有的输入信息，输出信息应该有相似的稀疏度，并且有足够的置 1 比特处理噪声和子采样
 */
#pragma once
#include "type.hpp"
#include <vector>
#include <map>
#include <string>

using namespace std;
 /** @brief
  * 将不同类型的输入数据编码成HTM适用的稀疏分布表征（SDR）。
  * 编码器负责确定哪些输出比特应该是 1，哪些应该是 0，从而捕获数据的重要语义特征。
  * 相似的输入信息应该产生彼此高度重叠的 SDR。
  */
class ScalarEncoder {
public:
    ScalarEncoder() = default;
    /**
     * 构造函数
     *
     * @param w 编码单个数值需要的比特数 -- 输出信号的“width”（一般不少于20~25）
     * @param minValue 序列右区间
     * @param maxValue 序列左区间
     * @param bucketNum 桶的数量
     * @param clipInput 当输入超过区间，若为true截断数值，否则不做处理
     */
    ScalarEncoder(int w, double minValue, double maxValue, int bucketNum, bool clipInput);


    /**
     * @brief 通过配置文件构造标量编码器
     */
    ScalarEncoder(const map<string, string>& config);

    ~ScalarEncoder() = default;

    void init(int w, double minValue, double maxValue, int bucketNum, bool clipInput);

    /**
     * @brief 将输入数值的编码结果写入output
     * @param input 需要编码的标量
     * @param output 需要提前分配好足够的空间（n_）
     */
    void encodeIntoArray(double input, UInt output[]);

    double decode(UInt bucketIdx);

    /**
     * @brief 获取输出数据长度
     * @return 每编码一个数据需要的bit数量
     */
    UInt getOutputWidth() const {
        return n_;
    }

    /**
     * @return 返回当前的bucket数量
     */
    UInt bucketNum() const {
        return bucketNum_;
    }

    /**
     * @return 最近编码的一个数据对应的bucket索引
     */
    UInt lastBucketIdx() const {
        return lastBucketIdx_;
    }
private:
    int w_;
    double minValue_;
    double maxValue_;
    UInt bucketNum_;
    UInt n_;
    bool clipInput_;
    UInt lastBucketIdx_;
}; // end class ScalarEncoder

