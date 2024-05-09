/**
 * @file 文件声明并实现了一些辅助函数，用于方便系统的使用
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
 * @brief 返回arr的上下界，上界保存在max，下界保存在min
 * @param beg 需要获得最大最小值的开始迭代器
 * @param end 需要获得最大最小值的结束迭代器
 * @param min 存储最小值
 * @param max 存储最大值
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
 * @brief 分行打印元素，可以选择每行打印的元素数量
 * @param beg 需要打印数组的开始迭代器
 * @param end 需要打印数组的结束迭代器
 * @param n 用于控制每行打印的元素数量
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
 * @brief 读取配置文件，读取的字段以及内容都为std::string，可能还需要
 * 进一步处理。
 * @param path 需要读取的配置文件的地址
 * @return config
 */
config readConfig(const string& path);

/**
 * @brief 根据给定输入数据路径将数据读入返回数组
 * @param path 输入数据的路径
 * @return 包含文件内所有数据的数组
 */
vector<Real64> getDataVector(const string& path);

/**
 * @brief 根据配置文件获取输入流
 * @return 输入流
 */
istream& getInputStream(const map<string, string>& config);

/** 
 * @brief 将字符串中的无符号整数读取到数组中
 * @param str 需要读取的字符串
 * @return 包含字符串中所有数据的数组
 */
vector<UInt> stov(const string& str);
