/** @file ʵ���˱�����������
 * ����������
 * �ڱ�������ʱ����һЩ��Ҫ�ķ�����Ҫ���ǣ�
 * 1. �������Ƶ�����Ӧ�ò�����Ծ���ر˴��ص��� SDR
 * 2. ��ͬ����������Ӧ�����ǲ�����ͬ��������
 * 3. �����е�������Ϣ�������ϢӦ������ͬ��ά�ȣ�����������
 * 4. �����е�������Ϣ�������ϢӦ�������Ƶ�ϡ��ȣ��������㹻���� 1 ���ش����������Ӳ���
 */
#pragma once
#include "type.hpp"
#include <vector>
#include <map>
#include <string>

using namespace std;
 /** @brief
  * ����ͬ���͵��������ݱ����HTM���õ�ϡ��ֲ�������SDR����
  * ����������ȷ����Щ�������Ӧ���� 1����ЩӦ���� 0���Ӷ��������ݵ���Ҫ����������
  * ���Ƶ�������ϢӦ�ò����˴˸߶��ص��� SDR��
  */
class ScalarEncoder {
public:
    ScalarEncoder() = default;
    /**
     * ���캯��
     *
     * @param w ���뵥����ֵ��Ҫ�ı����� -- ����źŵġ�width����һ�㲻����20~25��
     * @param minValue ����������
     * @param maxValue ����������
     * @param bucketNum Ͱ������
     * @param clipInput �����볬�����䣬��Ϊtrue�ض���ֵ������������
     */
    ScalarEncoder(int w, double minValue, double maxValue, int bucketNum, bool clipInput);


    /**
     * @brief ͨ�������ļ��������������
     */
    ScalarEncoder(const map<string, string>& config);

    ~ScalarEncoder() = default;

    void init(int w, double minValue, double maxValue, int bucketNum, bool clipInput);

    /**
     * @brief ��������ֵ�ı�����д��output
     * @param input ��Ҫ����ı���
     * @param output ��Ҫ��ǰ������㹻�Ŀռ䣨n_��
     */
    void encodeIntoArray(double input, UInt output[]);

    double decode(UInt bucketIdx);

    /**
     * @brief ��ȡ������ݳ���
     * @return ÿ����һ��������Ҫ��bit����
     */
    UInt getOutputWidth() const {
        return n_;
    }

    /**
     * @return ���ص�ǰ��bucket����
     */
    UInt bucketNum() const {
        return bucketNum_;
    }

    /**
     * @return ��������һ�����ݶ�Ӧ��bucket����
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

