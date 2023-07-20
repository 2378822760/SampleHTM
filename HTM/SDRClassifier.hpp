/**
 * @file ʵ����һ��ר�õ�softmax�����磬���÷��򴫲��㷨�Ÿ���Ȩ�أ�
 * û��ƫ�ýڵ㡣
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

    /** ���캯��
     * @brief ����һ��SDR��������Ϊÿһ��Ԥ�ⲽ����������һ��������softmax���硣
     * @param steps Ԥ�ⲽ����֧�ֶ��ͬʱԤ����������
     * @param cellsNum �趨��ʱ���(TM)�е�ϸ����������softmax������Ϊ�����ڵ������
     * @param bucketNum �趨�ڱ�����(Encoder)��Ͱ����������softmax������Ϊ����ڵ�ĸ�����
     * @param lr ���򴫲�ʱ��ѧϰ�ʡ�
     */
    SDRClassifier(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum,
        Real64 lr);

    SDRClassifier(const map<string, string>& config);

    ~SDRClassifier() = default;

    void init(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum,
        Real64 lr);

    /** SDRClassifier����Ҫ���к�����
     * @brief ����TM���õļ���ϸ����Ϊÿһ����������Ԥ������
     * @param patternNZ ��ʱ���(TM)����ó��ļ���ϸ��������
     * @param actualValIdx ��ʵֵ��Encoder�ж�Ӧ��Ͱ��������
     * @param learn Ϊtrueʱ�����·��򴫲���
     * @param result �ɲ���������map��ÿ��������Ӧһ�����飬���鳤��ΪEncoder��
     * Ͱ�������������i���ʾ�ɵ�ǰpatternNZȷ��Encoder�е�i��Ͱ�ĸ��ʡ�
     * @return ����ֵ������result�С�
     */
    void compute(const vector<UInt>& patternNZ, UInt actualValIdx, bool learn,
        ClassifierResult& result);

    /**
     * @brief ��ÿ��������ӡÿ��Ȩֵ��
     */
    void logWeight() const;

    /**
     * @brief ��ÿ��������ӡÿ�������
     */
    void logResult(ClassifierResult& result) const;

private:
    /**
     * @brief softmax�����ǰ�򴫲���Ϊÿ��������Ӧ�����綼���м��㡣
     * ���㹫ʽ��softmax(x �� weight)������ʾ��ˡ�
     * @param patternNZ ��ʱ���(TM)����ó��ļ���ϸ��������
     * @param result �ɲ���������map��ÿ��������Ӧһ�����飬���鳤��ΪEncoder��
     * Ͱ�������������i���ʾ�ɵ�ǰpatternNZȷ��Encoder�е�i��Ͱ�ĸ��ʡ�
     * @return ����ֵ������result�С�
     */
    void infer_(const vector<UInt>& patternNZ, ClassifierResult& result);

    /**
     * @brief softmax����ķ��򴫲���Ϊÿ���Ϸ�������Ӧ�����綼���м��㡣
     * Ȩֵ���¹�ʽ��w[i][j] -= dL/dw[i][j] * lr
     * @param onehot ��ʵֵ��Ӧ�Ľ���Ķ��ȱ���(ֻ��һ��bitΪ1,
     * ����Ϊ0�Ķ����ƴ��䳤��ΪEncoder��Ͱ������)��
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
