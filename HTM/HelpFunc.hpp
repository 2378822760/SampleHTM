/**
 * @file �ļ�������ʵ����һЩ�������������ڷ���ϵͳ��ʹ��
 */
#pragma once
#include "type.hpp"
#include "Encoder.hpp"
#include "SDRClassifier.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

using namespace std;
/**
 * @brief ����arr�����½磬�Ͻ籣����max���½籣����min
 * @param beg ��Ҫ��������Сֵ�Ŀ�ʼ������
 * @param end ��Ҫ��������Сֵ�Ľ���������
 * @param min �洢��Сֵ
 * @param max �洢���ֵ
 */
template<typename T, typename iterationType>
void getArrayRange(iterationType beg, iterationType end, T& min, T& max) {
	min = max = *beg;
	size_t arraySize = 0;
	for (iterationType it = beg; it != end; it++) {
		arraySize++;
		T tmp = *it;
		min = min > tmp ? tmp : min;
		max = max < tmp ? tmp : max;
	}
	max += (max - min) / (arraySize - 1);
}


/**
 * @brief ���д�ӡԪ�أ�����ѡ��ÿ�д�ӡ��Ԫ������
 * @param beg ��Ҫ��ӡ����Ŀ�ʼ������
 * @param end ��Ҫ��ӡ����Ľ���������
 * @param n ���ڿ���ÿ�д�ӡ��Ԫ������
 */
template<typename iterationType>
void printCodedDate(iterationType beg, iterationType end, UInt n) {
	UInt cnt = 0;
	for (iterationType it = beg; it != end; it++) {
		cout << *it << ' ';
		cnt++;
        if (cnt % n == 0) cout << endl;
	}
}


/**
 * @brief ��ȡ�����ļ�����ȡ���ֶ��Լ����ݶ�Ϊstd::string�����ܻ���Ҫ
 * ��һ������
 * @param path ��Ҫ��ȡ�������ļ��ĵ�ַ
 * @return config
 */
config readConfig(const string& path);

/**
 * @brief ���ݸ�����������·�������ݶ��뷵������
 * @param path �������ݵ�·��
 * @return �����ļ����������ݵ�����
 */
vector<Real64> getDataVector(const string& path);

/**
 * @brief ���������ļ���ȡ������
 * @return ������
 */
istream& getInputStream(const map<string, string>& config);

/** 
 * @brief ���ַ����е��޷���������ȡ��������
 * @param str ��Ҫ��ȡ���ַ���
 * @return �����ַ������������ݵ�����
 */
vector<UInt> stov(const string& str);
