/**
 * @file ʵ����һ��ר�õ�softmax�����磬���÷��򴫲��㷨�Ÿ���Ȩ�أ�
 * û��ƫ�ýڵ㡣
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
 * @brief ��������Ϊkey�洢���н��
*/
typedef map<UInt, Eigen::MatrixXd> ClassifierResult;

/**
 * @brief һ��Softmax���������������
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
    /** ���캯��
     * @brief ����һ��SDR��������Ϊÿһ��Ԥ�ⲽ����������һ��������softmax���硣
     * @param steps Ԥ�ⲽ����֧�ֶ��ͬʱԤ����������
     * @param cellsNum �趨��ʱ���(TM)�е�ϸ����������softmax������Ϊ�����ڵ������
     * @param bucketNum �趨�ڱ�����(Encoder)��Ͱ����������softmax������Ϊ����ڵ�ĸ�����
     * @param lr ���򴫲�ʱ��ѧϰ�ʡ�
     */
    SDRClassifier(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum, Real64 lr);

    /**
     * @brief ���������ļ���ʼ��
     * @param config �������ļ���ȡ��SDRClassifier��Ӧ�������
    */
    SDRClassifier(const map<string, string>& config);

    ~SDRClassifier() = default;

    void init(const vector<UInt>& steps, UInt cellsNum, UInt bucketNum, Real64 lr);

    /** SDRClassifier����Ҫ���к�����
     * @brief ����TM���õļ���ϸ����Ϊÿһ����������Ԥ������
     * @param patternNZ ��ʱ���(TM)����ó��ļ���ϸ��������
     * @param actualValIdx ��ʵֵ��Encoder�ж�Ӧ��Ͱ��������
     * @param learn Ϊtrueʱ�����·��򴫲���
     * @return �ɲ���������map��ÿ��������Ӧһ�����飬���鳤��ΪEncoder��
     * Ͱ�������������i���ʾ�ɵ�ǰpatternNZȷ��Encoder�е�i��Ͱ�ĸ��ʡ�
     */
    ClassifierResult compute(const vector<UInt>& patternNZ, UInt actualValIdx, bool learn);
private:
    /**
     * @brief softmax�����ǰ�򴫲���Ϊÿ��������Ӧ�����綼���м��㡣
     * ���㹫ʽ��softmax(x �� weight)������ʾ��ˡ�
     * @param inputHistory ���뵽SDRClassifier�е���ʷ���ݣ���ౣ��maxStep����ʷ���롣
     * @param steps ��Ҫ�����ƶϵĲ����б�
     * @param networks �����б�����Ӧ��Softmax�����б�
     * @return �ɲ���������map��ÿ��������Ӧһ�����飬���鳤��ΪEncoder��
     * Ͱ�������������i���ʾ�ɵ�ǰpatternNZȷ��Encoder�е�i��Ͱ�ĸ��ʡ�
     */
    ClassifierResult infer_(const deque<Eigen::MatrixXd>& inputHistory, const vector<UInt>& steps,
        map<UInt, Classifier>& networks, bool learn);

    /**
     * @brief softmax����ķ��򴫲���Ϊÿ���Ϸ�������Ӧ�����綼���м��㡣
     * @param y ���������Ӧ����ʵֵ����СΪbucketNum
     * @param steps ��Ҫ�����ƶϵĲ����б�
     * @param networks �����б�����Ӧ��Softmax�����б�
     * @param inputHistory ���뵽SDRClassifier�е���ʷ���ݣ���ౣ��maxStep����ʷ���롣
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
